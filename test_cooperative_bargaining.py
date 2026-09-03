import unittest
from types import SimpleNamespace

import numpy as np

from DGSolver import (
    DGSolver,
    _solve_sampled_terminal_gamma_sequence,
    filter_monotonic_cost_candidates,
    select_convex_cost_result,
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


class ConvexCostSelectionTests(unittest.TestCase):
    def test_selects_minimum_weighted_player_cost(self):
        candidates = [
            (0, 0.2, 1.0, 9.0),
            (1, 0.5, 4.0, 4.0),
            (2, 0.8, 8.0, 1.0),
        ]

        selected = select_convex_cost_result(candidates, (0.25, 0.75))

        self.assertEqual(selected[:2], (2, 0.8))

    def test_rejects_nonconvex_weights(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            select_convex_cost_result([(0, 0.5, 1.0, 2.0)], (1.0, 1.0))

    def test_solver_rejects_unknown_selection_method(self):
        game = GameDynamics(
            0.1,
            np.zeros(4),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            dynamics_type=1,
        )
        with self.assertRaisesRegex(ValueError, "cooperative_selection"):
            DGSolver(
                game,
                x1f=np.zeros((1, 2)),
                x2f=np.zeros((1, 2)),
                cooperative_selection="unknown",
            )


class NashBargainingFallbackTests(unittest.TestCase):
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


class LearnedPlayer2ActionTests(unittest.TestCase):
    def test_cooperative_mode_does_not_read_saved_player2_action(self):
        class AnalyzedData:
            @property
            def u2(self):
                raise AssertionError("cooperative mode read the saved u2 action")

        solver = DGSolver.__new__(DGSolver)
        solver.cooperative = True
        solver.LearnedData = SimpleNamespace(AnalyzedData=AnalyzedData())

        self.assertIsNone(solver._learned_player2_action(np.array([1.0])))

    def test_non_cooperative_mode_preserves_saved_player2_action(self):
        solver = DGSolver.__new__(DGSolver)
        solver.cooperative = False
        solver.LearnedData = SimpleNamespace(
            AnalyzedData=SimpleNamespace(
                u2=np.array([[1.0, 2.0], [3.0, 4.0]])
            )
        )

        learned_action = solver._learned_player2_action(np.array([0.25, 0.75]))

        np.testing.assert_allclose(learned_action, [2.5, 3.5])


class BackupControllerTests(unittest.TestCase):
    def test_update_splices_solution_to_matching_raw_safe_trajectory(self):
        terminal_state = np.array([1.0, 0.0, 2.0, 0.0])
        raw_data = SimpleNamespace(
            t=[4.0, 5.0, 6.0, 7.0],
            x=[
                np.array([0.0, 0.0, 3.0, 0.0]),
                terminal_state.copy(),
                np.array([1.5, 0.0, 2.5, 0.0]),
                np.array([2.0, 0.0, 3.0, 0.0]),
            ],
            u=[
                np.array([9.0, 9.0, 9.0, 9.0]),
                np.array([1.0, 0.0, 1.0, 0.0]),
                np.array([1.0, 0.0, 1.0, 0.0]),
                np.zeros(4),
            ],
            p1_total_cost=20.0,
            p2_total_cost=30.0,
        )
        solver = DGSolver.__new__(DGSolver)
        solver.dt = 0.5
        solver.N = 2
        solver.game = SimpleNamespace(nx=4, nu=4)
        solver.LearnedData = SimpleNamespace(RawData=[raw_data])
        solution = SimpleNamespace(
            t=10.0,
            x1=np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
            x2=np.array([[1.0, 0.0], [1.5, 0.0], [2.0, 0.0]]),
            u1=np.array([[1.0, 0.0], [1.0, 0.0]]),
            u2=np.array([[1.0, 0.0], [1.0, 0.0]]),
            terminal_sample_state=terminal_state,
            terminal_sample_time=5.0,
            player1_cost=7.0,
            player2_cost=8.0,
        )

        solver.backup_controller_update(solution)

        self.assertEqual(solver.backup.x.shape, (5, 4))
        self.assertEqual(solver.backup.u.shape, (5, 4))
        np.testing.assert_allclose(solver.backup.time, [10.0, 10.5, 11.0, 12.0, 13.0])
        np.testing.assert_allclose(solver.backup.x[2], terminal_state)
        np.testing.assert_allclose(solver.backup.x[3:], raw_data.x[2:])
        np.testing.assert_allclose(solver.backup.u[2:], raw_data.u[1:])
        self.assertEqual(solver.backup.cost1, 7.0)
        self.assertEqual(solver.backup.cost2, 8.0)

    def test_update_rejects_terminal_state_missing_from_raw_data(self):
        solver = DGSolver.__new__(DGSolver)
        solver.dt = 1.0
        solver.N = 1
        solver.game = SimpleNamespace(nx=4, nu=4)
        solver.LearnedData = SimpleNamespace(
            RawData=[
                SimpleNamespace(
                    t=[0.0],
                    x=[np.zeros(4)],
                    u=[np.zeros(4)],
                )
            ]
        )
        solution = SimpleNamespace(
            t=0.0,
            x1=np.zeros((2, 2)),
            x2=np.zeros((2, 2)),
            u1=np.zeros((1, 2)),
            u2=np.zeros((1, 2)),
            terminal_sample_state=np.ones(4),
        )

        with self.assertRaisesRegex(ValueError, "not found"):
            solver.backup_controller_update(solution)

    def test_backup_controller_tracks_forward_without_rewinding(self):
        solver = DGSolver.__new__(DGSolver)
        solver.N = 2
        solver.dt = 1.0
        solver.game = SimpleNamespace(nx=4, nu=4, nx1=2, nu1=2)
        solver.proximity_Q = np.eye(4)
        solver.Solution = SimpleNamespace(success=True)
        solver.backup = SimpleNamespace(
            time=np.array([0.0, 1.0, 2.0]),
            x=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [2.0, 0.0, 2.0, 0.0],
                ]
            ),
            u=np.array(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [2.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            indx=0,
        )

        control = solver.backup_controller(np.array([1.1, 0.0, 1.1, 0.0]))
        np.testing.assert_allclose(control, [2.0, 0.0, 2.0, 0.0])
        self.assertEqual(solver.backup.indx, 1)
        np.testing.assert_allclose(
            solver.Solution.x1,
            [[1.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
        )
        np.testing.assert_allclose(
            solver.Solution.x2,
            [[1.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
        )
        np.testing.assert_allclose(
            solver.Solution.u1,
            [[2.0, 0.0], [0.0, 0.0]],
        )
        np.testing.assert_allclose(
            solver.Solution.u2,
            [[2.0, 0.0], [0.0, 0.0]],
        )
        np.testing.assert_allclose(solver.Solution.x0, [1.1, 0.0, 1.1, 0.0])
        self.assertEqual(solver.Solution.indx, 0)
        self.assertEqual(solver.Solution.t, 1.0)
        self.assertEqual(solver.Solution.status, "backup_controller")
        self.assertTrue(solver.Solution.is_backup)
        self.assertEqual(
            solver.Solution.x1.shape[0],
            solver.Solution.u1.shape[0] + 1,
        )

        # A later call cannot replay the already-consumed control at index 0.
        control = solver.backup_controller(np.zeros(4))
        np.testing.assert_allclose(control, [2.0, 0.0, 2.0, 0.0])
        self.assertEqual(solver.backup.indx, 1)
        self.assertTrue(solver.Solution.used_backup_controller)
        self.assertFalse(solver.Solution.success)

    def test_step_uses_backup_when_direct_solve_fails(self):
        solver = DGSolver.__new__(DGSolver)
        solver.constraint_mode = "convex_hull"
        solver.N = 2
        solver.dt = 1.0
        solver.game = SimpleNamespace(nx=4, nu=4, nx1=2, nu1=2)
        solver.proximity_Q = np.eye(4)
        solver.Solution = SimpleNamespace(success=True)
        solver.backup = SimpleNamespace(
            time=np.array([0.0]),
            x=np.zeros((1, 4)),
            u=np.array([[0.25, 0.0, -0.25, 0.0]]),
            indx=0,
        )

        def failed_solve(*args, **kwargs):
            solver.last_solve_success = False
            return np.full(4, 99.0)

        solver._step_once = failed_solve
        control = solver.step(0.0, np.zeros(4))

        np.testing.assert_allclose(control, [0.25, 0.0, -0.25, 0.0])
        self.assertTrue(solver.Solution.used_backup_controller)


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


class ThreePlayerGameTests(unittest.TestCase):
    def setUp(self):
        self.targets = [
            np.array([[1.0, -1.0, 0.0, 0.0]]),
            np.array([[-1.0, 1.0, 0.0, 0.0]]),
            np.array([[0.0, -1.0, 0.0, 0.0]]),
        ]
        self.x0 = np.array([
            -1.0, 1.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
        ])

    def test_dynamics_integrates_all_three_players(self):
        game = GameDynamics(
            0.1, self.x0, *self.targets, dynamics_type=2
        )
        game.reset_game()
        control = np.array([1.0, 0.0, 0.0, 1.0, -1.0, 0.0])
        self.assertEqual(game.step(control), game.STEP_OK)
        self.assertEqual(game.get_state().shape, (12,))
        np.testing.assert_allclose(game.get_state()[2::4], [0.1, 0.0, -0.1])

    def test_solver_builds_three_player_kkt_blocks(self):
        game = GameDynamics(
            0.1, self.x0, *self.targets, dynamics_type=2
        )
        solver = DGSolver(
            game, *self.targets, horizon=2, cooperative=True,
            alpha=(0.5, 0.25),
            bargaining_gammas=((0.2, 0.3), (0.5, 0.25)),
        )
        backend = solver.build_solver()
        self.assertEqual(backend.params["nx"], 12)
        self.assertEqual(len(backend.Z_len), 6)
        self.assertEqual(solver.cooperative_cost_weights.shape, (3,))
        np.testing.assert_allclose(solver.alpha_vec[0], [0.5, 0.25])
        np.testing.assert_allclose(
            1.0 - np.sum(solver.alpha_vec, axis=1), 0.25
        )

    def test_solver_rejects_alpha_pair_outside_simplex(self):
        game = GameDynamics(0.1, self.x0, *self.targets, dynamics_type=2)
        with self.assertRaisesRegex(ValueError, r"alpha1 \+ alpha2 <= 1"):
            DGSolver(
                game, *self.targets, cooperative=True,
                bargaining_gammas=((0.7, 0.4),),
            )


class UnicycleDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.x1f = np.array([[2.0, 0.0, 0.0, 0.0]])
        self.x2f = np.array([[-2.0, 0.0, 0.0, np.pi]])

    def test_rk4_integrates_unicycle_state(self):
        game = GameDynamics(
            0.1,
            np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5, np.pi]),
            self.x1f, self.x2f, dynamics_type=3,
        )
        game.reset_game()
        self.assertEqual(game.step(np.array([1.0, 0.0, 0.0, 0.0])), game.STEP_OK)
        np.testing.assert_allclose(game.x[:4], [0.105, 0.0, 1.1, 0.0], atol=1e-10)

    def test_rejects_turn_rate_and_lateral_acceleration_violations(self):
        initial = np.array([0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.5, np.pi])
        game = GameDynamics(
            0.1, initial, self.x1f, self.x2f, dynamics_type=3,
            psi_dot_max=1.0, an_max=1.0,
        )
        game.reset_game()
        self.assertEqual(
            game.step(np.array([0.0, 0.6, 0.0, 0.0])),
            game.INPUT_OUTSIDE_BOUNDS,
        )
        np.testing.assert_allclose(game.x, initial)
        self.assertEqual(
            game.step(np.array([0.0, 1.1, 0.0, 0.0])),
            game.INPUT_OUTSIDE_BOUNDS,
        )

    def test_solver_uses_nonlinear_unicycle_dynamics(self):
        game = GameDynamics(
            0.1,
            np.array([0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, np.pi]),
            self.x1f, self.x2f, dynamics_type=3,
        )
        solver = DGSolver(game, self.x1f, self.x2f, horizon=2)
        backend = solver.build_solver()
        self.assertEqual(backend.params["dynamics_type"], 3)
        self.assertGreater(backend.J.nnz_out(0), 0)


if __name__ == "__main__":
    unittest.main()
