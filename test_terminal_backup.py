import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from DGSolver import DGSolver
from Game import GameDynamics


class TerminalBackupTests(unittest.TestCase):
    def make_solver(self, players=2, mode="sampled_points"):
        targets = [np.array([[2.0 * p, 0.0]]) for p in range(players)]
        final = np.concatenate([target.ravel() for target in targets])
        offset = np.tile([0.2, 0.0], players)
        game = GameDynamics(0.1, final - offset, *targets, dynamics_type=1)
        game.reset_game()
        solver = DGSolver(game, *targets, horizon=2, constraint_mode=mode)
        states = np.array([final - offset, final - offset / 2, final])
        controls = np.array([np.tile([1.0, 0.0], players)] * 2
                            + [np.zeros(game.nu)])
        solver.backup = SimpleNamespace(time=np.arange(3) * 0.1,
                                        x=states, u=controls, indx=0)
        solver.Solution = SimpleNamespace(success=True, indx=0)
        solver.last_solve_success = True
        for player in range(players):
            setattr(solver.Solution, f"x{player + 1}",
                    states[:, 2 * player:2 * player + 2].copy())
        return solver

    def test_terminal_prediction_skips_solving_and_recovery_on_subsequent_steps(self):
        for players in (2, 3):
            for mode in ("sampled_points", "convex_hull"):
                with self.subTest(players=players, mode=mode):
                    solver = self.make_solver(players, mode)
                    with patch.object(solver, '_step_once', side_effect=AssertionError("new solve")), \
                            patch.object(solver, '_step_over_sampled_terminal_states',
                                         side_effect=AssertionError("terminal search")):
                        for index in (1, 2):
                            control = solver.step_with_recovery(
                                index * 0.1, solver.backup.x[index])
                            np.testing.assert_array_equal(control, solver.backup.u[index])
                            self.assertEqual(solver.Solution.backup_index, index)
                            self.assertTrue(solver.Solution.used_backup_controller)
                            self.assertFalse(solver.last_solve_success)
                            self.assertEqual(solver.N, 2)

    def test_direct_step_checks_inside_horizon_with_existing_arrival_tolerance(self):
        solver = self.make_solver()
        for player, target in enumerate(solver.targets):
            prediction = getattr(solver.Solution, f'x{player + 1}')
            prediction[1] = target + [np.sqrt(solver.proximity_minval) / 2, 0.0]
            prediction[-1] = target + [0.2, 0.0]
        with patch.object(solver, '_step_once', side_effect=AssertionError("new solve")):
            control = solver.step(0.0, solver.backup.x[0])
        np.testing.assert_array_equal(control, solver.backup.u[0])

    def test_solving_continues_without_an_accepted_joint_terminal_prediction(self):
        for scenario in ('one_player_away', 'different_arrival_steps',
                         'failed_solution', 'no_solution', 'no_backup'):
            with self.subTest(scenario=scenario):
                solver = self.make_solver(3)
                if scenario == 'one_player_away':
                    solver.Solution.x3[-1, 0] += 0.2
                elif scenario == 'different_arrival_steps':
                    solver.Solution.x3[1] = solver.targets[2]
                    solver.Solution.x3[-1, 0] += 0.2
                elif scenario == 'failed_solution':
                    solver.Solution.success = False
                elif scenario == 'no_solution':
                    solver.Solution = SimpleNamespace(success=False)
                elif scenario == 'no_backup':
                    del solver.backup
                expected = np.full(solver.game.nu, 0.5)
                with patch.object(solver, '_step_once', return_value=expected) as solve:
                    control = solver.step(0.1, solver.game.x)
                solve.assert_called_once()
                np.testing.assert_array_equal(control, expected)
                self.assertFalse(solver._terminal_backup_active)


if __name__ == '__main__':
    unittest.main()
