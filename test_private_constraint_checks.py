import unittest

import casadi as ca
import numpy as np

from Game import GameDynamics


class PrivateConstraintChecksTests(unittest.TestCase):
    def make_game(self, mode=4, players=3):
        nx = {1: 2, 2: 4, 3: 3, 4: 4}[mode]
        targets = [np.array([[2.0 * p, 0.0, *([0.0] * (nx - 2))]])
                   for p in range(players)]
        game = GameDynamics(0.1, np.concatenate(targets).ravel(), *targets,
                            dynamics_type=mode)
        game.reset_game()
        return game

    def test_private_function_is_the_only_source_of_private_bounds(self):
        for mode in (1, 2, 3, 4):
            for players in (2, 3):
                with self.subTest(mode=mode, players=players):
                    game = self.make_game(mode, players)
                    # Remove private bounds to check that step() does not keep
                    # hidden position, speed, acceleration, or steering checks.
                    game.f_private = lambda x, u: 1.0
                    game.f_shared = lambda *args: 1.0
                    game.x[:] = 100.0
                    self.assertEqual(game.step(np.full(game.nu, 20.0)), game.STEP_OK)

    def test_rate_violation_is_rejected_for_each_player_before_integration(self):
        for players in (2, 3):
            for player in range(players):
                game = self.make_game(players=players)
                initial = game.x.copy()
                control = np.zeros(game.nu)
                control[player * 2 + 1] = game.psidot_max + 0.1
                self.assertEqual(game.step(control), game.PRIVATE_CONSTRAINT_VIOLATED)
                np.testing.assert_array_equal(game.x, initial)
                self.assertEqual(game.t, 0.0)
                self.assertEqual(game.history['status'][-1], game.PRIVATE_CONSTRAINT_VIOLATED)

    def test_custom_state_constraint_is_checked_after_integration(self):
        game = self.make_game(mode=1, players=2)
        x, u = ca.SX.sym('x', 2), ca.SX.sym('u', 2)
        game.f_private = ca.Function('limited_y', [x, u], [0.05 - x[1]])
        self.assertEqual(game.step([0.0, 1.0, 0.0, 0.0]), game.PRIVATE_CONSTRAINT_VIOLATED)
        self.assertAlmostEqual(game.t, game.dt)
        self.assertAlmostEqual(game.x[1], 0.1)
        self.assertEqual(game.history['status'][-1], game.PRIVATE_CONSTRAINT_VIOLATED)

    def test_scalar_vector_and_multiple_outputs_use_existing_tolerance(self):
        for output in ('scalar', 'vector', 'multiple'):
            for residual, valid in ((0.0, True), (-0.004, True), (-0.006, False)):
                with self.subTest(output=output, residual=residual):
                    game = self.make_game()
                    x, u = ca.SX.sym('x', 4), ca.SX.sym('u', 2)
                    outputs = {'scalar': [residual],
                               'vector': [ca.DM([1.0, residual])],
                               'multiple': [1.0, ca.DM([1.0, residual])]}[output]
                    game.f_private = ca.Function('custom_private', [x, u], outputs)
                    expected = game.STEP_OK if valid else game.PRIVATE_CONSTRAINT_VIOLATED
                    self.assertEqual(game.step(np.zeros(game.nu)), expected)

    def test_nonfinite_input_state_and_constraint_residual_are_rejected(self):
        for source in ('input', 'state', 'residual'):
            game = self.make_game()
            control = np.zeros(game.nu)
            if source == 'input':
                control[-1] = np.nan
            elif source == 'state':
                game.x[-1] = np.inf
            else:
                game.f_private = lambda x, u: np.nan
            self.assertEqual(game.step(control), game.PRIVATE_CONSTRAINT_VIOLATED)
            self.assertEqual(game.t, 0.0)

    def test_shared_constraints_are_still_enforced(self):
        game = self.make_game()
        game.f_private = lambda x, u: 1.0
        game.f_shared = lambda *args: -1.0
        self.assertEqual(game.step(np.zeros(game.nu)), game.SHARED_CONSTRAINT_VIOLATED)


if __name__ == '__main__':
    unittest.main()
