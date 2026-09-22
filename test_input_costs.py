import unittest

import casadi as ca
import numpy as np

from DGSolver import DGSolver
from Game import GameDynamics


class InputCostTests(unittest.TestCase):
    def test_unicycle_penalizes_only_acceleration_for_every_player(self):
        for players in (2, 3):
            with self.subTest(players=players):
                targets = [np.zeros((1, 3)) for _ in range(players)]
                game = GameDynamics(0.1, np.zeros(players * 3), *targets, dynamics_type=3)
                solver = DGSolver(game, *targets, R1=0.2, R2=0.3, R3=0.4)
                for player, cost in enumerate(solver.stage_costs):
                    weight = (solver.R1, solver.R2, solver.R3)[player]
                    for heading in (-np.pi, -0.5, 0.0, 1.2, np.pi):
                        self.assertAlmostEqual(float(cost(np.zeros(3), [0.4, heading])), weight * 0.4**2)
                        self.assertEqual(float(cost(np.zeros(3), [0.0, heading])), 0.0)
                    u = ca.SX.sym('u', 2)
                    hessian = ca.Function('input_hessian', [u], [ca.hessian(cost(np.zeros(3), u), u)[0]])
                    np.testing.assert_allclose(hessian([0.4, 1.2]), np.diag([2 * weight, 0.0]))
                self.assertAlmostEqual(float(solver.l1(np.zeros(3), [0.4, 1.2], np.zeros(3), [0.7, -2.0])), 0.2 * 0.4**2)
                self.assertAlmostEqual(float(solver.l2(np.zeros(3), [0.7, -2.0], np.zeros(3), [0.4, 1.2])), 0.3 * 0.7**2)
                if players == 3:
                    self.assertAlmostEqual(float(solver.l3(np.zeros(3), [0.4, 1.2])), 0.4 * 0.4**2)

    def test_integrators_still_penalize_both_inputs(self):
        for mode, nx in ((1, 2), (2, 4)):
            with self.subTest(dynamics_type=mode):
                targets = [np.zeros((1, nx)) for _ in range(3)]
                game = GameDynamics(0.1, np.zeros(3 * nx), *targets, dynamics_type=mode)
                solver = DGSolver(game, *targets, R1=0.2, R2=0.3, R3=0.4)
                for player, cost in enumerate(solver.stage_costs):
                    weight = (solver.R1, solver.R2, solver.R3)[player]
                    self.assertAlmostEqual(float(cost(np.zeros(nx), [0.4, 1.2])), weight * (0.4**2 + 1.2**2))


if __name__ == '__main__':
    unittest.main()
