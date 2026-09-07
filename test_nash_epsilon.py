import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from DGSolver import DGSolver, NASH_IMPROVEMENT_EPSILON, _nash_product
from Game import GameDynamics
from LDG_Simulation_aux import init_learned_data


class NashEpsilonTests(unittest.TestCase):
    def test_zero_improvements_have_positive_factors(self):
        epsilon = NASH_IMPROVEMENT_EPSILON
        self.assertEqual(_nash_product([0.0, 0.0, 2.0]), epsilon**2 * (2.0 + epsilon))
        self.assertEqual(_nash_product([-0.05, 2.0]), epsilon * (2.0 + epsilon))

    def test_sample_selection_improves_remaining_players(self):
        cases = [
            [(10.0, 9.0), (10.0, 2.0)],
            [(9.0, 10.0), (2.0, 10.0)],
            [(10.0, 9.0, 9.0), (10.0, 2.0, 3.0)],
            [(10.0, 10.0, 9.0), (10.0, 10.0, 2.0)],
            [(10.0, 9.0, 10.0), (10.0, 2.0, 10.0)],
            [(9.0, 10.0, 10.0), (2.0, 10.0, 10.0)],
            [(9.0, 9.0, 10.0), (2.0, 3.0, 10.0)],
        ]
        for candidates in cases:
            with self.subTest(costs=candidates):
                players = len(candidates[0])
                mode, nx = (3, 3) if players == 2 else (2, 4)
                targets = [np.array([[float(p), 0.0] + [0.0] * (nx - 2)])
                           for p in range(players)]
                initial = np.concatenate([target.ravel() for target in targets])
                game = GameDynamics(0.1, initial, *targets, dynamics_type=mode)
                gammas = [0.2, 0.4] if players == 2 else [[0.2, 0.2], [0.4, 0.2]]
                solver = DGSolver(game, *targets, cooperative=True, horizon=1,
                                  bargaining_gammas=gammas)
                solver.LearnedData = init_learned_data()
                data = solver.LearnedData.AnalyzedData
                data.t = [0.1]
                data.state = np.array([initial])
                data.Cost2Go, data.Cost2Go2 = [0.0], [0.0]
                data.Cost2Go3 = [0.0] if players == 3 else []
                data.n_data = 1
                calls = []

                def solve(*args, **kwargs):
                    costs = candidates[len(calls)]
                    calls.append(kwargs['forced_alpha'])
                    solver.Solver = object()
                    solver.last_solve_success = True
                    solver.Solution = SimpleNamespace(sigma=np.ones(1), costs=costs)
                    for player in range(players):
                        setattr(solver.Solution, f'u{player + 1}', np.zeros((1, 2)))

                with patch.object(solver, 'calc_a_set', return_value=(None, 0.0)), \
                        patch.object(solver, '_step_once', side_effect=solve), \
                        patch.object(solver, '_player1_cost', side_effect=lambda s, _: s.costs[0]), \
                        patch.object(solver, '_player2_cost', side_effect=lambda s, _: s.costs[1]), \
                        patch.object(solver, '_player3_cost', side_effect=lambda s, _: s.costs[2]), \
                        patch.object(solver, 'backup_controller_update'):
                    solver._step_over_sampled_terminal_states(
                        0.0, initial, use_all_terminal_points=True,
                        disagreement_costs=[10.0] * players,
                        previous_iteration_costs=[10.0] * players,
                    )
                self.assertEqual(solver.Solution.costs, candidates[1])
                np.testing.assert_array_equal(solver.Solution.bargaining_gamma, gammas[1])
                improvements = 10.0 - np.array(candidates[1])
                np.testing.assert_array_equal(solver.Solution.bargaining_improvements, improvements)
                self.assertEqual(solver.Solution.nash_product, _nash_product(improvements))
                self.assertGreater(solver.Solution.nash_product, 0.0)


if __name__ == '__main__':
    unittest.main()
