import unittest
from types import SimpleNamespace

import numpy as np

from DGSolver import select_nash_bargaining_result
from LDG_Simulation_aux import (
    init_learned_data,
    rebuild_analyzed_data,
    remaining_cost_budget,
)


class NashBargainingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
