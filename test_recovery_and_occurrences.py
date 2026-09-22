import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from DGSolver import DGSolver
from Game import GameDynamics
from LDG_Simulation_aux import init_learned_data, rebuild_analyzed_data


class TerminalOccurrenceTests(unittest.TestCase):
    def make_solver(self, players=2):
        q = np.array([[2.0 * p, 0.0] for p in range(players)]).ravel()
        move = np.tile([1.0, 0.0], players)
        delta = 0.1 * move
        targets = [np.array([(q + 2 * delta)[2 * p:2 * p + 2]]) for p in range(players)]
        game = GameDynamics(0.1, q - delta, *targets, dynamics_type=1)
        game.reset_game()
        solver = DGSolver(game, *targets, horizon=1, cooperative=True,
                          bargaining_gammas=[0.5] if players == 2 else [[1/3, 1/3]])
        learned = init_learned_data()
        learned.RawData = [
            SimpleNamespace(t=np.arange(3) * 0.1, x=np.array([q, q + delta, q + 2 * delta]),
                            u=np.array([move, move, np.zeros(game.nu)])),
            SimpleNamespace(t=np.arange(5) * 0.1, x=np.array([q, q - delta, q, q + delta, q + 2 * delta]),
                            u=np.array([-move, move, move, move, np.zeros(game.nu)])),
        ]
        rebuild_analyzed_data(learned, 1, game, solver)
        solver.LearnedData = learned
        solution = SimpleNamespace(t=0.0, success=True, indx=0, sigma=np.ones(1),
                                   ai_xf_vec=np.zeros((0, 1)), terminal_sample_state=q,
                                   terminal_sample_time=0.0)
        for player in range(players):
            setattr(solution, f'x{player + 1}', np.array([(q - delta)[2*player:2*player+2], q[2*player:2*player+2]]))
            setattr(solution, f'u{player + 1}', np.array([[1.0, 0.0]]))
        return solver, solution

    def test_selected_older_occurrence_keeps_its_own_suffix_and_costs(self):
        for players in (2, 3):
            solver, solution = self.make_solver(players)

            def solve(*args, terminal_learned_data, **kwargs):
                occurrence = tuple(terminal_learned_data.AnalyzedData.occurrences[0])
                solver.last_solve_success = occurrence == (0, 0)
                solver.Solver = object()
                if solver.last_solve_success:
                    solver.Solution = copy.deepcopy(solution)

            with patch.object(solver, 'calc_a_set', return_value=(None, 0.0)), \
                    patch.object(solver, '_step_once', side_effect=solve):
                solver.step(0.0, solver.game.x, use_all_terminal_points=True)
            self.assertEqual(solver.Solution.terminal_occurrence, (0, 0))
            self.assertEqual(solver.backup.terminal_occurrence, (0, 0))
            np.testing.assert_array_equal(solver.backup.u[solver.N:], solver.LearnedData.RawData[0].u)
            for player in range(players):
                cost = sum(float(solver.stage_costs[player](x[2*player:2*player+2], u[2*player:2*player+2]))
                           for x, u in zip(solver.backup.x, solver.backup.u))
                self.assertAlmostEqual(cost, getattr(solver.Solution, f'player{player + 1}_cost'))

    def test_missing_or_invalid_identity_never_silently_picks_a_different_suffix(self):
        solver, solution = self.make_solver()
        with self.assertRaisesRegex(ValueError, 'ambiguous terminal occurrence'):
            solver.backup_controller_update(solution)
        solution.terminal_occurrence = (0, 99)
        with self.assertRaisesRegex(ValueError, 'not found'):
            solver.backup_controller_update(solution)

    def test_rebuild_records_absolute_indices_and_excludes_unfinished_iteration(self):
        solver, _ = self.make_solver()
        solver.LearnedData.RawData.append(SimpleNamespace())
        rebuild_analyzed_data(solver.LearnedData, 1, solver.game, solver, iterations_to_use=1)
        self.assertEqual(solver.LearnedData.AnalyzedData.occurrences, [(1, i) for i in range(5)])

    def test_backup_initialization_skips_current_incomplete_iteration(self):
        for players in (2, 3):
            solver, _ = self.make_solver(players)
            completed = solver.LearnedData.RawData[-1]
            solver.LearnedData.RawData.append(SimpleNamespace(t=[0.0], x=[solver.game.x],
                                                              u=[np.zeros(solver.game.nu)], p1_total_cost=None))
            solver.game.iteration = 2
            retry = DGSolver(solver.game, *[np.array([t]) for t in solver.targets],
                             LearnedData=solver.LearnedData, horizon=3, cooperative=True)
            np.testing.assert_array_equal(retry.backup.x, completed.x)
            np.testing.assert_array_equal(retry.backup.u, completed.u)


class RecoveryTests(unittest.TestCase):
    def run_recovery(self, succeed_at, players=2):
        solver, _ = TerminalOccurrenceTests().make_solver(players)
        solver.backup_controller_init()
        original_backup = copy.deepcopy(solver.backup)
        original_horizon = solver.N
        calls, accepted = [], {}
        budget = tuple([20.0] * players)

        def step(instance, t, x0, **kwargs):
            calls.append((instance.N, kwargs.get('use_all_terminal_points', False)))
            self.assertEqual(kwargs['previous_iteration_costs'], budget)
            instance.last_solve_success = len(calls) == succeed_at
            instance.Solution = SimpleNamespace(success=instance.last_solve_success, indx=0,
                                                is_backup=not instance.last_solve_success)
            if instance is not solver:
                # Failed trials must not mutate the retained backup or its cache.
                instance.backup.u[:] = 0.75
                instance._sampled_solver_cache['retry'] = instance.N
            if instance.last_solve_success:
                instance.Solver = SimpleNamespace(horizon=instance.N)
                instance.solver = instance.Solver
                instance.is_built = True
                accepted['solver'] = instance.Solver
                accepted['backup'] = instance.backup
            return np.full(instance.game.nu, instance.N if instance.last_solve_success else -1.0)

        with patch.object(DGSolver, 'step', step):
            control = solver.step_with_recovery(0.0, solver.game.x, previous_iteration_costs=budget)
        return solver, original_backup, original_horizon, control, calls, accepted

    def test_zero_backup_index_triggers_expanded_search_and_adopts_whole_longer_solver(self):
        for players in (2, 3):
            solver, _, horizon, control, calls, accepted = self.run_recovery(4, players)
            self.assertEqual(calls, [(horizon, False), (horizon, True), (horizon+1, True), (horizon+2, True)])
            self.assertEqual(solver.N, horizon + 2)
            self.assertEqual(solver.alpha_vec.shape[0], solver.N + 1)
            self.assertIs(solver.Solver, accepted['solver'])
            self.assertIs(solver.backup, accepted['backup'])
            self.assertTrue(solver.is_built)
            self.assertTrue(solver.last_solve_success)
            np.testing.assert_array_equal(control, np.full(solver.game.nu, solver.N))

    def test_all_failed_retries_preserve_original_horizon_and_backup(self):
        solver, backup, horizon, control, calls, _ = self.run_recovery(None)
        self.assertEqual(len(calls), 5)
        self.assertEqual(solver.N, horizon)
        np.testing.assert_array_equal(solver.backup.u, backup.u)
        np.testing.assert_array_equal(control, np.full(solver.game.nu, -1.0))
        self.assertNotIn('retry', solver._sampled_solver_cache)

    def test_success_stops_retries_immediately(self):
        for succeed_at in (1, 2):
            solver, _, horizon, _, calls, _ = self.run_recovery(succeed_at)
            self.assertEqual(len(calls), succeed_at)
            self.assertEqual(solver.N, horizon)

    def test_simulation_still_requests_joint_control_when_player_one_has_arrived(self):
        # Execute the real simulation's bootstrap/learning dispatch in isolation.
        tree = ast.parse(Path(__file__).with_name('LDG_Simulation.py').read_text())
        dispatch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                        and isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name) and node.test.left.id == 'iter'
                        and isinstance(node.test.ops[0], ast.Eq)
                        and isinstance(node.test.comparators[0], ast.Constant)
                        and node.test.comparators[0].value == 0)
        code = compile(ast.Module(body=[dispatch], type_ignores=[]), 'simulation_dispatch', 'exec')
        for players in (2, 3):
            solver, _ = TerminalOccurrenceTests().make_solver(players)
            solver.game.x[:2] = solver.targets[0]
            joint_control = np.array([0.0, 0.0] + [1.0, 0.0] * (players - 1))
            solver.step_with_recovery = Mock(return_value=joint_control)
            namespace = dict(iter=1, Game=solver.game, Solver1=solver, player_count=players,
                             current_cost1=0.0, current_cost2=0.0, current_cost3=0.0,
                             active_disagreement_costs=None, previous_iteration_costs=None,
                             np=np)
            exec(code, namespace)
            solver.step_with_recovery.assert_called_once()
            np.testing.assert_array_equal(namespace['u1'], joint_control)


if __name__ == '__main__':
    unittest.main()
