import unittest
from unittest.mock import patch

import numpy as np

from DGSolver import DGSolver
from Game import GameDynamics
from LDG_Simulation_aux import init_learned_data


class CandidateCaptured(Exception):
    pass


class TerminalSamplingTests(unittest.TestCase):
    def make_solver(self, players, cooperative=True):
        mode = 3 if players == 2 else 2
        state_size = 3 if mode == 3 else 4
        targets = [np.array([[float(p), 0.0] + [0.0] * (state_size - 2)])
                   for p in range(players)]
        initial = np.concatenate([target.ravel() for target in targets])
        game = GameDynamics(0.1, initial, *targets, dynamics_type=mode, v_min=0.0)
        learned = init_learned_data()
        analyzed = learned.AnalyzedData
        analyzed.t = np.arange(6) * 0.1
        analyzed.state = np.tile(initial, (6, 1))
        analyzed.Cost2Go = np.arange(6.0)
        analyzed.Cost2Go2 = np.arange(6.0) + 10.0
        analyzed.u2 = np.arange(12.0).reshape(6, 2)
        if players == 3:
            analyzed.Cost2Go3 = np.arange(6.0) + 20.0
        analyzed.n_data = 6
        solver = DGSolver(game, *targets, LearnedData=learned, horizon=1,
                          cooperative=cooperative)
        return solver, initial

    def test_sample_five_slices_only_participating_players_costs(self):
        for players, cooperative in ((2, True), (2, False), (3, True)):
            with self.subTest(players=players, cooperative=cooperative):
                solver, initial = self.make_solver(players, cooperative)
                with patch.object(solver, 'calc_a_set', return_value=(None, 0.0)), \
                        patch.object(solver, '_step_once', side_effect=CandidateCaptured) as solve:
                    with self.assertRaises(CandidateCaptured):
                        solver._step_over_sampled_terminal_states(
                            0.6, initial, use_all_terminal_points=True,
                        )
                candidate = solve.call_args.kwargs['terminal_learned_data'].AnalyzedData
                self.assertEqual(candidate.n_data, 1)
                np.testing.assert_allclose(candidate.Cost2Go, [5.0])
                np.testing.assert_allclose(candidate.Cost2Go2, [15.0])
                np.testing.assert_allclose(candidate.Cost2Go3, [25.0] if players == 3 else [])
                if not cooperative:
                    np.testing.assert_allclose(candidate.u2, [[10.0, 11.0]])
                self.assertEqual(solver.LearnedData.AnalyzedData.n_data, 6)

    def test_two_player_time_filter_ignores_empty_third_player_costs(self):
        solver, initial = self.make_solver(2)
        with patch.object(solver, '_learned_player2_action', return_value=None), \
                patch.object(solver, 'build_solver', side_effect=CandidateCaptured) as build:
            with self.assertRaises(CandidateCaptured):
                solver._step_once(0.4, initial, precomputed_a_set=(None, 0.0))
        candidate = build.call_args.kwargs['Terminal_Safe_Set']
        np.testing.assert_allclose(candidate.Cost2Go, [5.0])
        np.testing.assert_allclose(candidate.Cost2Go3, [])


if __name__ == '__main__':
    unittest.main()
