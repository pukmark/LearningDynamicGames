import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from DGSolver import DGSolver
from Game import GameDynamics


class BackupInitialGuessTests(unittest.TestCase):
    def make_solver(self, players):
        mode = 3 if players == 2 else 2
        extra = [0.5] if mode == 3 else [0.5, 0.0]
        targets = [np.array([[float(p), 1.0, *extra]]) for p in range(players)]
        initial = np.concatenate([target.ravel() for target in targets])
        game = GameDynamics(0.1, initial, *targets, dynamics_type=mode, W=10.0)
        solver = DGSolver(game, *targets, horizon=2, cooperative=True)
        states = [initial]
        controls = np.array([
            np.concatenate([[0.1 * (k + 1), 0.2 * (p + 1)] for p in range(players)])
            for k in range(5)
        ])
        for control in controls[:-1]:
            states.append(np.concatenate([
                np.asarray(game.dynamics_fun(
                    states[-1][p * game.nx1:(p + 1) * game.nx1],
                    control[p * game.nu1:(p + 1) * game.nu1],
                )).ravel() for p in range(players)
            ]))
        solver.backup = SimpleNamespace(
            time=np.arange(5) * 0.1, x=np.array(states), u=controls, indx=0,
        )
        # Start one step into the backup: the horizon ends at row 3, not 4.
        x0 = solver.backup.x[1].copy()
        terminal = SimpleNamespace(AnalyzedData=SimpleNamespace(
            state=solver.backup.x[[3]].copy(), Cost2Go=np.array([1.0]),
            Cost2Go2=np.array([2.0]), Cost2Go3=np.array([3.0]), n_data=1,
        ))
        return solver, x0, terminal

    def capture_guess(self, solver, x0, terminal, **kwargs):
        main = SimpleNamespace()
        with patch('DGSolver._ensure_julia'), patch('DGSolver.Main', main, create=True), \
                patch('DGSolver.jl', SimpleNamespace(eval=Mock(
                    side_effect=RuntimeError('guess captured'))), create=True):
            with self.assertRaisesRegex(RuntimeError, 'guess captured'):
                solver._step_once(
                    0.1, x0, terminal_learned_data=terminal,
                    precomputed_a_set=(None, 0.0), **kwargs,
                )
        self.assertEqual(main.z0.size, solver.Solver.Z.numel())
        states, controls = [], []
        offset = 0
        for lengths in solver.Solver.Z_len[:solver.game.n_players]:
            x_len, u_len = lengths[:2]
            states.append(main.z0[offset:offset + x_len].reshape(
                solver.N + 1, solver.game.nx1, order='F'))
            offset += x_len
            controls.append(main.z0[offset:offset + u_len].reshape(
                solver.N, solver.game.nu1, order='F'))
            offset += u_len + sum(lengths[2:])
        return np.concatenate(states, axis=1), np.concatenate(controls, axis=1)

    def test_matching_terminal_reuses_backup_in_both_solvers_and_cached_backends(self):
        for players in (2, 3):
            with self.subTest(players=players):
                solver, x0, terminal = self.make_solver(players)
                original = copy.deepcopy(solver.backup)
                original_solution = solver.Solution
                for cached in (False, True):
                    kwargs = {'terminal_solver': solver.Solver} if cached else {}
                    states, controls = self.capture_guess(solver, x0, terminal, **kwargs)
                    np.testing.assert_allclose(states, original.x[1:4])
                    np.testing.assert_allclose(controls, original.u[1:3])
                    self.assertEqual(solver.backup.indx, 0)
                    self.assertIs(solver.Solution, original_solution)
                    np.testing.assert_array_equal(solver.backup.x, original.x)
                    np.testing.assert_array_equal(solver.backup.u, original.u)

    def test_other_players_terminal_speed_must_also_match(self):
        for players in (2, 3):
            with self.subTest(players=players):
                solver, x0, terminal = self.make_solver(players)
                terminal.AnalyzedData.state[0, -1] += 0.2
                states, controls = self.capture_guess(solver, x0, terminal)
                self.assertFalse(np.allclose(controls, solver.backup.u[1:3]))
                np.testing.assert_array_equal(states[0], x0)

    def test_missing_backup_uses_existing_guess(self):
        for players in (2, 3):
            with self.subTest(players=players):
                solver, x0, terminal = self.make_solver(players)
                del solver.backup
                states, controls = self.capture_guess(solver, x0, terminal)
                self.assertTrue(np.all(np.isfinite(states)))
                self.assertTrue(np.all(np.isfinite(controls)))
                np.testing.assert_array_equal(states[0], x0)

    def test_explicit_two_player_control_guess_is_preserved(self):
        solver, x0, terminal = self.make_solver(2)
        controls1 = np.full((solver.N, 2), 0.05)
        controls2 = np.full((solver.N, 2), 0.15)
        _, controls = self.capture_guess(
            solver, x0, terminal, u1_0=controls1, u2_0=controls2,
        )
        np.testing.assert_array_equal(controls, np.c_[controls1, controls2])

    def test_matching_uses_tolerance_and_does_not_alias_stored_trajectory(self):
        solver, x0, terminal = self.make_solver(2)
        terminal.AnalyzedData.state += 1e-10
        x0[0] += 1e-4
        states, controls = solver._backup_initial_guess(x0, terminal.AnalyzedData)
        np.testing.assert_array_equal(states[0], x0)
        self.assertFalse(np.shares_memory(states, solver.backup.x))
        self.assertFalse(np.shares_memory(controls, solver.backup.u))
        self.assertEqual(solver.backup.indx, 0)

    def test_forward_selection_and_padding_match_backup_controller(self):
        solver, _, terminal = self.make_solver(3)
        solver.backup.indx = 3
        terminal.AnalyzedData.state = solver.backup.x[[-1]].copy()
        # An old state must not rewind the controller past its current index.
        x0 = solver.backup.x[0].copy()
        states, controls = solver._backup_initial_guess(x0, terminal.AnalyzedData)
        np.testing.assert_array_equal(states[0], x0)
        np.testing.assert_array_equal(states[1:], np.repeat(solver.backup.x[[-1]], 2, axis=0))
        np.testing.assert_array_equal(controls, solver.backup.u[3:5])
        self.assertEqual(solver.backup.indx, 3)
        solver.backup_controller(x0)
        for p in range(3):
            np.testing.assert_array_equal(
                controls[:, p * 2:(p + 1) * 2], getattr(solver.Solution, f'u{p + 1}'),
            )


if __name__ == '__main__':
    unittest.main()
