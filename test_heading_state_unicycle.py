import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import casadi as ca
import numpy as np

from DGSolver import DGSolver
from Game import GameDynamics
from LDG_Simulation_aux import player_state


class HeadingStateUnicycleTests(unittest.TestCase):
    def make_game(self, players=2, **limits):
        initial = np.array([[-2.0, 0.0, 0.5, 0.2],
                            [0.0, 2.0, 0.5, -1.5],
                            [2.0, 0.0, 0.5, 2.8]])[:players]
        targets = [np.array([[0.0, float(p), 0.0, 0.0]]) for p in range(players)]
        game = GameDynamics(0.1, initial.ravel(), *targets, dynamics_type=4,
                            v_min=0.0, L=20.0, W=20.0, **limits)
        game.reset_game()
        return game, targets

    def capture_guess(self, solver, state, terminal=None):
        main = SimpleNamespace()
        with patch('DGSolver._ensure_julia'), patch('DGSolver.Main', main, create=True), \
                patch('DGSolver.jl', SimpleNamespace(eval=Mock(
                    side_effect=RuntimeError('guess captured'))), create=True):
            with self.assertRaisesRegex(RuntimeError, 'guess captured'):
                solver._step_once(0.0, state, precomputed_a_set=(None, 0.0),
                                  terminal_learned_data=terminal)
        states, controls, offset = [], [], 0
        for lengths in solver.Solver.Z_len[:solver.game.n_players]:
            x_len, u_len = lengths[:2]
            states.append(main.z0[offset:offset + x_len].reshape(solver.N + 1, 4, order='F'))
            offset += x_len
            controls.append(main.z0[offset:offset + u_len].reshape(solver.N, 2, order='F'))
            offset += u_len + sum(lengths[2:])
        return np.concatenate(states, axis=1), np.concatenate(controls, axis=1)

    def test_numpy_and_symbolic_dynamics_use_state_heading(self):
        for players in (2, 3):
            game, _ = self.make_game(players)
            controls = np.tile([0.7, -0.3], players)
            states = game.x.reshape(players, 4)
            expected = np.column_stack((0.5 * np.cos(states[:, 3]),
                                        0.5 * np.sin(states[:, 3]),
                                        np.full(players, 0.7), np.full(players, -0.3))).ravel()
            np.testing.assert_allclose(game.dynamics(game.x, controls), expected)
            np.testing.assert_allclose(game.dynamics(game.x[:4], controls[:2]), expected[:4])
            for symbol in (ca.SX, ca.MX):
                x, u = symbol.sym('x', game.nx), symbol.sym('u', game.nu)
                dynamics = ca.Function('derivative', [x, u], [game.dynamics(x, u)])
                np.testing.assert_allclose(np.asarray(dynamics(game.x, controls)).ravel(), expected)

    def test_rk4_follows_constant_rate_circular_motion(self):
        game, _ = self.make_game()
        initial = game.x.copy()
        control = np.tile([0.0, 0.5], 2)
        expected = initial.reshape(2, 4).copy()
        psi = expected[:, 3].copy()
        expected[:, 0] += np.sin(psi + 0.05) - np.sin(psi)
        expected[:, 1] += np.cos(psi) - np.cos(psi + 0.05)
        expected[:, 3] += 0.05
        self.assertEqual(game.step(control), game.STEP_OK)
        np.testing.assert_allclose(game.x, expected.ravel(), atol=2e-10, rtol=0)
        np.testing.assert_allclose(np.asarray(game.dynamics_fun(initial[:4], control[:2])).ravel(),
                                   game.x[:4], atol=1e-12)

    def test_rate_bounds_in_simulation_and_symbolic_constraints(self):
        for lower, upper in ((-0.5, 0.5), (-0.2, 0.35)):
            game, _ = self.make_game(pdot_min=lower, psidot_max=upper)
            for rate in (lower, upper):
                game.reset_game()
                self.assertGreaterEqual(np.asarray(game.f_private(game.x[:4], [0.0, rate])).min(), 0)
                self.assertEqual(game.step(np.tile([0.0, rate], 2)), game.STEP_OK)
            for rate in (lower - 0.01, upper + 0.01):
                game.reset_game()
                initial = game.x.copy()
                self.assertLess(np.asarray(game.f_private(game.x[:4], [0.0, rate])).min(), 0)
                self.assertEqual(game.step(np.tile([0.0, rate], 2)), game.INPUT_OUTSIDE_BOUNDS)
                np.testing.assert_array_equal(game.x, initial)
                self.assertEqual(game.t, 0.0)

    def test_invalid_rate_bounds_and_explicit_initial_heading(self):
        for lower, upper in ((0.5, -0.5), (0.0, 0.0), (-np.inf, 0.5), (-0.5, np.nan)):
            with self.assertRaisesRegex(ValueError, 'heading-rate bounds'):
                self.make_game(pdot_min=lower, psidot_max=upper)
        self.assertEqual(player_state(1, 2, vx=0.7, dynamics_type=4, psi=1.2), [1, 2, 0.7, 1.2])
        self.assertEqual(player_state(1, 2, vx=0.7, dynamics_type=3), [1, 2, 0.7])

    def test_bootstrap_turns_in_place_and_tracks_final_heading(self):
        game, _ = self.make_game(pdot_min=-0.2, psidot_max=0.35)
        game.x[:4] = [0.0, 0.0, 0.0, np.pi]
        acceleration, rate = game._unicycle_goal_controller(0, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(acceleration, 0.0)
        self.assertAlmostEqual(rate, game.pdot_min)
        game.x[:4] = [0.0, 0.0, 0.0, 0.0]
        acceleration, rate = game._unicycle_goal_controller(0, [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(acceleration, 0.0)
        self.assertEqual(rate, game.psidot_max)
        # Crossing the angle branch cut should produce a small positive turn.
        self.assertAlmostEqual(game.heading_rate_control(np.pi - 0.01, -np.pi + 0.01, 1.0), 0.02)

    def test_both_solver_builds_and_initial_guesses_use_bounded_rate(self):
        for players in (2, 3):
            game, targets = self.make_game(players, pdot_min=-0.2, psidot_max=0.35)
            solver = DGSolver(game, *targets, horizon=5, cooperative=True)
            states, controls = self.capture_guess(solver, game.x)
            self.assertEqual(solver.Solver.params['dynamics_type'], 4)
            self.assertEqual(solver.Solver.params['nx'], 4 * players)
            self.assertGreater(solver.Solver.J.nnz_out(0), 0)
            np.testing.assert_array_equal(states[0], game.x)
            self.assertTrue(np.all(controls[:, 1::2] >= game.pdot_min))
            self.assertTrue(np.all(controls[:, 1::2] <= game.psidot_max))
            for k in range(solver.N):
                for player in range(players):
                    x, u = states[k, 4 * player:4 * player + 4], controls[k, 2 * player:2 * player + 2]
                    np.testing.assert_allclose(states[k + 1, 4 * player:4 * player + 4],
                                               np.asarray(game.dynamics_fun(x, u)).ravel())
                    self.assertGreaterEqual(np.asarray(game.f_private(x, u)).min(), -1e-10)
            for cost in solver.stage_costs:
                self.assertAlmostEqual(float(cost(game.x[:4], [0.4, -0.2])),
                                       float(cost(game.x[:4], [0.4, 0.35])))

    def test_matching_terminal_reuses_four_state_backup(self):
        for players in (2, 3):
            game, targets = self.make_game(players)
            solver = DGSolver(game, *targets, horizon=2, cooperative=True)
            controls = np.tile([0.1, 0.2], (3, players))
            states = [game.x.copy()]
            for control in controls[:2]:
                states.append(np.concatenate([
                    np.asarray(game.dynamics_fun(states[-1][4 * p:4 * p + 4],
                                                 control[2 * p:2 * p + 2])).ravel()
                    for p in range(players)]))
            solver.backup = SimpleNamespace(time=np.arange(3) * game.dt, x=np.array(states),
                                            u=controls, indx=0)
            terminal = SimpleNamespace(AnalyzedData=SimpleNamespace(
                state=np.array(states)[[-1]], Cost2Go=np.ones(1), Cost2Go2=np.ones(1),
                Cost2Go3=np.ones(1), n_data=1))
            guess_x, guess_u = self.capture_guess(solver, game.x, terminal)
            np.testing.assert_allclose(guess_x, states)
            np.testing.assert_allclose(guess_u, controls[:2])


if __name__ == '__main__':
    unittest.main()
