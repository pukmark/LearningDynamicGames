import unittest
from types import SimpleNamespace

import numpy as np

from DGSolver import (
    _solve_sampled_terminal_gamma_sequence,
    filter_monotonic_cost_candidates,
    select_nash_bargaining_result,
    solution_has_no_interaction,
)
from Game import GameDynamics
from LDG_Simulation_aux import (
    init_learned_data,
    rebuild_analyzed_data,
    remaining_cost_budget,
)


class NashBargainingTests(unittest.TestCase):
    def test_candidates_cannot_worsen_either_previous_iteration_total(self):
        candidates = [
            (0, 0.25, 50.0, 30.0),
            (1, 0.50, 39.0, 41.0),
            (2, 0.75, 39.0, 30.0),
        ]
        acceptable = filter_monotonic_cost_candidates(
            candidates,
            executed_costs=(10.0, 20.0),
            previous_iteration_costs=(50.0, 55.0),
        )
        self.assertEqual([result[0] for result in acceptable], [2])

    def test_zero_sigma_stops_remaining_gammas_for_terminal_state(self):
        calls = []

        class FakeSolver:
            sigma_zero_tolerance = 1e-8
            Solver = None
            Solution = SimpleNamespace()

            def _step_once(self, *args, forced_alpha=None, **kwargs):
                calls.append(forced_alpha)
                sigma = np.array([1.0]) if forced_alpha == 0.25 else np.zeros(2)
                self.Solution = SimpleNamespace(sigma=sigma)
                self.Solver = object()
                self.last_solve_success = True

            def _player1_cost(self, solution, candidate_data):
                return 1.0

            def _player2_cost(self, solution, candidate_data):
                return 2.0

        results = _solve_sampled_terminal_gamma_sequence(
            FakeSolver(), None, 3, 1, 3, 0.0, np.zeros(1),
            [0.25, 0.5, 0.75], None, None, None, 0.0,
        )

        self.assertEqual(calls, [0.25, 0.5])
        self.assertEqual(len(results), 2)
        self.assertTrue(results[-1][4].gamma_independent)
        np.testing.assert_allclose(
            results[-1][4].skipped_bargaining_gammas, [0.75]
        )

    def test_interaction_detection_uses_sigma_tolerance(self):
        self.assertTrue(
            solution_has_no_interaction(SimpleNamespace(sigma=[0.0, 1e-9]))
        )
        self.assertFalse(
            solution_has_no_interaction(SimpleNamespace(sigma=[0.0, 1e-4]))
        )

    def test_rejects_individually_unacceptable_candidates(self):
        candidates = [
            (0, 0.2, 2.0, 12.0),  # Player 2 rejects this candidate.
            (1, 0.5, 6.0, 6.0),
            (2, 0.8, 9.0, 4.0),
        ]
        selected = select_nash_bargaining_result(candidates, (10.0, 10.0))
        self.assertEqual(selected[:2], (1, 0.5))

    def test_returns_none_when_agreement_set_is_empty(self):
        candidates = [(0, 0.5, 11.0, 3.0), (1, 0.6, 3.0, 11.0)]
        self.assertIsNone(
            select_nash_bargaining_result(candidates, (10.0, 10.0))
        )

    def test_player2_chooses_when_player1_cannot_improve(self):
        candidates = [
            (0, 0.2, 10.0, 8.0),
            (1, 0.5, 10.0, 3.0),
            (2, 0.8, 10.0, 6.0),
        ]
        selected = select_nash_bargaining_result(candidates, (10.0, 10.0))
        self.assertEqual(selected[:2], (1, 0.5))

    def test_player1_chooses_when_player2_cannot_improve(self):
        candidates = [
            (0, 0.2, 7.0, 10.0),
            (1, 0.5, 2.0, 10.0),
            (2, 0.8, 5.0, 10.0),
        ]
        selected = select_nash_bargaining_result(candidates, (10.0, 10.0))
        self.assertEqual(selected[:2], (1, 0.5))


class LearnedCostToGoTests(unittest.TestCase):
    def test_remaining_budget_subtracts_executed_cost_for_each_player(self):
        budget = remaining_cost_budget((100.0, 80.0), (25.0, 30.0))
        np.testing.assert_allclose(budget, [75.0, 50.0])

    def test_rebuild_stores_both_players_cost_to_go(self):
        learned_data = init_learned_data()
        learned_data.RawData.append(
            SimpleNamespace(
                t=[0.0, 1.0],
                x=[np.array([0.0, 0.0, 1.0, 1.0]), np.ones(4)],
                u=[np.zeros(4), np.zeros(4)],
            )
        )
        game = SimpleNamespace(nx1=2, nu1=2)
        solver = SimpleNamespace(
            l1=lambda *_: 1.0,
            l2=lambda *_: 2.0,
            proximity_Q=np.eye(4),
        )

        rebuild_analyzed_data(learned_data, 0, game, solver)

        np.testing.assert_allclose(learned_data.AnalyzedData.Cost2Go, [2.0, 1.0])
        np.testing.assert_allclose(learned_data.AnalyzedData.Cost2Go2, [4.0, 2.0])


class SimpleControllerTests(unittest.TestCase):
    def test_player2_controller_is_symmetric_and_bounded(self):
        game = GameDynamics(
            0.1,
            np.array([-1.75, 1.5, 0.0, 0.0, 1.75, -1.5, 0.0, 0.0]),
            np.array([[1.5, -1.5, 0.0, 0.0]]),
            np.array([[-1.5, 1.5, 0.0, 0.0]]),
            dynamics_type=2,
        )
        game.reset_game()

        control1 = game.SimpleController1()
        control2 = game.SimpleController2()

        self.assertEqual(control1.shape, (game.nu1,))
        self.assertEqual(control2.shape, (game.nu2,))
        np.testing.assert_allclose(control2, -control1)
        self.assertTrue(np.all(control2 >= game.u_min))
        self.assertTrue(np.all(control2 <= game.u_max))

    def test_both_controllers_support_single_integrator_dynamics(self):
        game = GameDynamics(
            0.1,
            np.array([-1.75, 1.5, 1.75, -1.5]),
            np.array([[1.5, -1.5]]),
            np.array([[-1.5, 1.5]]),
            dynamics_type=1,
        )
        game.reset_game()

        self.assertEqual(game.SimpleController1().shape, (2,))
        self.assertEqual(game.SimpleController2().shape, (2,))


if __name__ == "__main__":
    unittest.main()
