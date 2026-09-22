import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from DGSolver import DGSolver
from Game import GameDynamics
from LDG_Simulation_aux import init_learned_data


class AcceptedPlanBaselineTests(unittest.TestCase):
    def make_solver(self, players, **kwargs):
        targets = [np.array([[2.0 * p, 2.0]]) for p in range(players)]
        initial = np.concatenate([target.ravel() - [0.0, 1.0] for target in targets])
        game = GameDynamics(0.1, initial, *targets, dynamics_type=1)
        solver = DGSolver(game, *targets, cooperative=True, horizon=1, **kwargs)
        learned = init_learned_data()
        raw = SimpleNamespace(t=[0.0, 0.1], x=np.array([initial, initial]),
                              u=np.zeros((2, game.nu)))
        for player in range(players):
            setattr(raw, f'p{player + 1}_total_cost', 100.0)
        learned.RawData = [raw]
        data = learned.AnalyzedData
        data.t = [0.1]
        data.state = np.array([initial])
        data.Cost2Go = data.Cost2Go2 = data.Cost2Go3 = [0.0]
        data.occurrences = [(0, 1)]
        data.n_data = 1
        solver.LearnedData = learned
        solver.backup_controller_init()
        return solver

    def search(self, solver, candidates, executed, **kwargs):
        players = solver.game.n_players
        solver.bargaining_gammas = (
            np.linspace(0.2, 0.8, len(candidates)) if players == 2
            else np.tile([0.3, 0.3], (len(candidates), 1))
        )
        calls = []

        def solve(*args, **options):
            costs = candidates[len(calls) % len(candidates)]
            calls.append(costs)
            solver.Solver = object()
            solver.last_solve_success = True
            solver.Solution = SimpleNamespace(
                costs=costs, sigma=np.ones(1), t=0.0, indx=0,
            )
            for player in range(players):
                setattr(solver.Solution, f'u{player + 1}', np.zeros((1, 2)))

        with ExitStack() as stack:
            stack.enter_context(patch.object(solver, 'calc_a_set', return_value=(None, 0.0)))
            stack.enter_context(patch.object(solver, '_prediction_reaches_target', return_value=False))
            stack.enter_context(patch.object(solver, '_step_once', side_effect=solve))
            stack.enter_context(patch.object(solver, 'backup_controller', return_value=np.zeros(solver.game.nu)))
            update = stack.enter_context(patch.object(solver, 'backup_controller_update'))
            for player in range(players):
                stack.enter_context(patch.object(
                    solver, f'_player{player + 1}_cost',
                    side_effect=lambda solution, _, p=player: solution.costs[p],
                ))
            solver.step_with_recovery(
                0.0, solver.game.x0,
                **{f'current_cost{p + 1}': executed[p] for p in range(players)},
                **kwargs,
            )
        return update, calls

    def test_bootstrap_baseline_advances_with_executed_cost(self):
        for players in (2, 3):
            solver = self.make_solver(players)
            np.testing.assert_allclose(solver._bargaining_baseline([0] * players), 100)
            np.testing.assert_allclose(solver._bargaining_baseline([10] * players), 90)

    def test_selected_plan_replaces_baseline_and_next_step_subtracts_only_new_cost(self):
        for players in (2, 3):
            solver = self.make_solver(players)
            # Comparing against a changing baseline would reject the second
            # candidate, although it has the greater product against 90.
            candidates = [[20.0, 80.0], [30.0, 30.0]]
            if players == 3:
                candidates = [costs + [30.0] for costs in candidates]
            update, _ = self.search(solver, candidates, [10] * players,
                                    previous_iteration_costs=[100] * players)
            update.assert_called_once()
            np.testing.assert_allclose(solver.Solution.disagreement_costs, 90)
            np.testing.assert_allclose(solver._bargaining_baseline([10] * players), 30)
            np.testing.assert_allclose(solver._bargaining_baseline([15] * players), 25)
            self.search(solver, [[20] * players], [15] * players,
                        previous_iteration_costs=[100] * players)
            np.testing.assert_allclose(solver.Solution.disagreement_costs, 25)
            np.testing.assert_allclose(solver._bargaining_baseline([17] * players), 18)

    def test_worse_plan_is_rejected_even_with_room_in_previous_iteration_budget(self):
        for players in (2, 3):
            solver = self.make_solver(players)
            self.search(solver, [[30] * players], [10] * players)
            totals = solver._accepted_plan_total_costs.copy()
            # One player worsens from 25 to 26, despite a much better sum.
            update, calls = self.search(solver, [[26] + [1] * (players - 1)],
                                        [15] * players,
                                        previous_iteration_costs=[100] * players)
            self.assertFalse(solver.last_solve_success)
            self.assertEqual(len(calls), 2)  # Normal and expanded recovery search.
            update.assert_not_called()
            np.testing.assert_array_equal(solver._accepted_plan_total_costs, totals)
            np.testing.assert_allclose(solver._bargaining_baseline([17] * players), 23)

    def test_previous_iteration_total_remains_an_independent_constraint(self):
        for players in (2, 3):
            solver = self.make_solver(players)
            update, _ = self.search(solver, [[50] * players], [10] * players,
                                    disagreement_costs=[100] * players,
                                    previous_iteration_costs=[55] * players)
            self.assertFalse(solver.last_solve_success)
            update.assert_not_called()
            np.testing.assert_allclose(solver._bargaining_baseline([10] * players), 90)

    def test_baseline_mode_changes_acceptance_after_an_improved_plan(self):
        for players in (2, 3):
            for mode in ('iteration_start', 'accepted_plan'):
                with self.subTest(players=players, mode=mode):
                    solver = self.make_solver(players, baseline_mode=mode)
                    self.search(solver, [[30] * players], [10] * players)
                    # Starting total is 100; the accepted plan's total is 40.
                    # With 15 executed, the remaining limits are 85 and 25.
                    update, _ = self.search(
                        solver, [[50] * players], [15] * players,
                        previous_iteration_costs=[100] * players,
                    )
                    if mode == 'iteration_start':
                        update.assert_called_once()
                        self.assertTrue(solver.last_solve_success)
                        np.testing.assert_allclose(solver.Solution.disagreement_costs, 85)
                        np.testing.assert_allclose(solver._accepted_plan_total_costs, 65)
                        np.testing.assert_allclose(solver._bargaining_baseline([20] * players), 80)
                    else:
                        update.assert_not_called()
                        self.assertFalse(solver.last_solve_success)
                        np.testing.assert_allclose(solver._accepted_plan_total_costs, 40)
                    np.testing.assert_allclose(solver._iteration_start_total_costs, 100)

    def test_iteration_start_rejects_candidates_above_starting_total(self):
        for players in (2, 3):
            solver = self.make_solver(players, baseline_mode='iteration_start')
            update, _ = self.search(solver, [[91] * players], [10] * players)
            update.assert_not_called()
            self.assertFalse(solver.last_solve_success)
            np.testing.assert_allclose(solver._bargaining_baseline([15] * players), 85)

    def test_iteration_start_can_initialize_from_previous_costs_without_backup(self):
        for players in (2, 3):
            solver = self.make_solver(players, baseline_mode='iteration_start')
            solver._iteration_start_total_costs = None
            solver._accepted_plan_total_costs = None
            self.assertIsNone(solver._bargaining_baseline([0] * players))
            previous = np.full(players, 100.0)
            np.testing.assert_allclose(solver._bargaining_baseline(
                [10] * players, previous_iteration_costs=previous), 90)
            previous[:] = 50  # The starting snapshot must not alias its input.
            self.search(solver, [[30] * players], [10] * players)
            np.testing.assert_allclose(solver._bargaining_baseline(
                [15] * players, previous_iteration_costs=previous), 85)

    def test_invalid_baseline_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'baseline_mode'):
            self.make_solver(2, baseline_mode='unknown')

    def test_fixed_override_remains_fixed_after_acceptance(self):
        for players in (2, 3):
            for mode in ('iteration_start', 'accepted_plan'):
                with self.subTest(players=players, mode=mode):
                    solver = self.make_solver(players, baseline_mode=mode)
                    solver.disagreement_costs = [80] * players
                    self.search(solver, [[30] * players], [10] * players)
                    np.testing.assert_allclose(solver._bargaining_baseline([15] * players), 80)
                    np.testing.assert_allclose(solver._bargaining_baseline(
                        [15] * players, disagreement_costs=[60] * players), 60)
                    solver.disagreement_costs = None
                    expected = 85 if mode == 'iteration_start' else 25
                    np.testing.assert_allclose(solver._bargaining_baseline([15] * players), expected)


if __name__ == '__main__':
    unittest.main()
