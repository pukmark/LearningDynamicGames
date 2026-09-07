import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from DGSolver import DGSolver
from Game import GameDynamics


class ThreePlayerInitialGuessTests(unittest.TestCase):
    def capture_guess(self, dynamics_type, terminal_positions=None):
        positions = np.array([[-2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
        goals = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        extra = {1: [], 2: [0.0, 0.0], 3: [0.5]}[dynamics_type]
        initial = np.array([np.r_[position, extra] for position in positions])
        targets = [np.array([np.r_[goal, extra]]) for goal in goals]
        game = GameDynamics(0.1, initial.ravel(), *targets, dynamics_type=dynamics_type,
                            L=20.0, W=20.0)
        game.reset_game()
        solver = DGSolver(game, *targets, horizon=10, cooperative=True)
        terminal = None
        if terminal_positions is not None:
            terminal = SimpleNamespace(AnalyzedData=SimpleNamespace(
                state=np.array([[value for position in terminal_positions
                                 for value in np.r_[position, extra]]]),
            ))
        main = SimpleNamespace()
        # Stop at the Julia boundary after constructing the actual packed z0.
        with patch('DGSolver._ensure_julia'), patch('DGSolver.Main', main, create=True), \
                patch('DGSolver.jl', SimpleNamespace(eval=Mock(
                    side_effect=RuntimeError('guess captured'))), create=True):
            with self.assertRaisesRegex(RuntimeError, 'guess captured'):
                solver._step_once_three_player(
                    0.0, initial.ravel(), precomputed_a_set=(None, 0.0),
                    terminal_learned_data=terminal,
                )
        np.testing.assert_array_equal(game.x, initial.ravel())
        self.assertEqual(main.z0.size, solver.Solver.Z.numel())
        guesses = []
        offset = 0
        for x_len, u_len, ai_len, slack_len in solver.Solver.Z_len[:3]:
            xp = main.z0[offset:offset + x_len].reshape(11, game.nx1, order='F')
            offset += x_len
            up = main.z0[offset:offset + u_len].reshape(10, 2, order='F')
            offset += u_len + ai_len + slack_len
            guesses.append((xp, up))
        return solver, initial, goals, guesses

    def test_guesses_make_progress_with_bounded_controls_and_consistent_dynamics(self):
        for mode in (1, 2, 3):
            with self.subTest(dynamics_type=mode):
                solver, initial, goals, guesses = self.capture_guess(mode)
                for player, (xp, up) in enumerate(guesses):
                    np.testing.assert_array_equal(xp[0], initial[player])
                    self.assertTrue(np.all(np.isfinite(xp)))
                    self.assertLess(np.linalg.norm(xp[-1, :2] - goals[player]),
                                    np.linalg.norm(xp[0, :2] - goals[player]))
                    for k in range(solver.N):
                        expected = solver._player_next_state(xp[k], up[k])
                        np.testing.assert_allclose(xp[k + 1], np.asarray(expected).ravel())
                        constraints = np.asarray(solver.game.f_private(xp[k], up[k]))
                        self.assertGreaterEqual(constraints.min(), -1e-10)
                    if mode == 3:
                        direction = goals[player] - initial[player, :2]
                        self.assertAlmostEqual(up[0, 1], np.arctan2(direction[1], direction[0]))
                        self.assertTrue(np.all(xp[:, 2] >= solver.game.v_min - 1e-10))
                        self.assertTrue(np.all(xp[:, 2] <= solver.game.v_max + 1e-10))

    def test_unicycle_aims_at_selected_terminal_sample(self):
        terminal_positions = np.array([[-3.0, -1.0], [1.0, 3.0], [3.0, 1.0]])
        _, initial, _, guesses = self.capture_guess(3, terminal_positions)
        for player, (xp, up) in enumerate(guesses):
            direction = terminal_positions[player] - initial[player, :2]
            self.assertAlmostEqual(up[0, 1], np.arctan2(direction[1], direction[0]))
            self.assertLess(np.linalg.norm(xp[-1, :2] - terminal_positions[player]),
                            np.linalg.norm(direction))


if __name__ == '__main__':
    unittest.main()
