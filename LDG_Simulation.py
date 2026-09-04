import os
os.system('clear')
import argparse
import numpy as np
import casadi as ca
import copy

from Game import GameDynamics
from DGSolver import DGSolver, initialize_pathsolver_runtime
from LDG_Simulation_aux import (
    append_terminal_learned_state,
    arrival_times,
    init_learned_data,
    is_shared_constraint_active,
    load_learned_data,
    player_state,
    rebuild_analyzed_data,
    record_learned_state,
    remaining_cost_budget,
    save_learned_data,
    should_reduce_alpha,
)
from SimulationPlot import *

np.random.seed(100)

L = 5.0
W = 4.0
dt = 0.1
tf = 15.0
dynamics_type = 2  # 1: single integrator, 2: double integrator, 3: unicycle
v_min = 0.1
v_max = 2.0
a_max = 2.0
psi_dot_max = 1.0  # rad/s
an_max = 1.0  # m/s^2, lateral acceleration |v * psi_dot|
terminal_constraint_mode = "sampled_points" # {"convex_hull", "sampled_points"}
# In cooperative mode Solver1 selects both the learned safe-set reconnection
# state and the shared-constraint equilibrium weight. The selection can use
# Nash bargaining or a convex weighted sum of the two players' costs-to-go.
cooperative_mode = True
bargaining_gammas = np.array([0.5])
bargaining_gamma1 = np.array([0.333, 0.4, 0.2])
bargaining_gamma2 = np.array([0.333, 0.2, 0.4])
cooperative_selection = "nash_bargaining" # "weighted_sum", "nash_bargaining"
cooperative_cost_weights = np.array([0.5, 0.5])
# Optional fixed (b1_t, b2_t) costs-to-go. When this is None, iterations after
# the bootstrap use the previous completed totals minus costs executed so far.
disagreement_costs = None
Niterations = 15
arrival_tolerance = 0.01
learned_data_path = "LearnedData.pkl"
x1f = np.array([player_state(1.5, -1.5, dynamics_type=dynamics_type)])
x2f = np.array([player_state(-1.75, 1.5, dynamics_type=dynamics_type)])
x3f = np.array([player_state(-0.5, -1.5, dynamics_type=dynamics_type)])
x0_players = (
    player_state(-1.75, 1.5, dynamics_type=dynamics_type),
    player_state(-0.5, -2.0, dynamics_type=dynamics_type),
    player_state(1.75, 1.5, dynamics_type=dynamics_type),
)
alpha1, alpha2 = 0.5, 0.25

max_workers = max(1, int(os.cpu_count() * 0.3))
# max_workers = 1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the learning dynamic-game simulation.")
    parser.add_argument(
        "--players", type=int, choices=(2, 3), default=3,
        help="number of players in the game (default: 2)",
    )
    parser.add_argument(
        "--cooperative", action="store_true",
        help="apply Solver1 jointly to both players",
    )
    parser.add_argument(
        "--bargaining-gammas", nargs="+", type=float, default=None,
        help="candidate alpha1 values for a two-player game",
    )
    parser.add_argument(
        "--bargaining-gamma1", nargs="+", type=float, default=None,
        help="candidate alpha1 values for a three-player game",
    )
    parser.add_argument(
        "--bargaining-gamma2", nargs="+", type=float, default=None,
        help="candidate alpha2 values paired with --bargaining-gamma1",
    )
    parser.add_argument(
        "--cooperative-selection",
        choices=("nash_bargaining", "weighted_sum"),
        default=cooperative_selection,
        help="select cooperative candidates by Nash bargaining or weighted cost",
    )
    parser.add_argument(
        "--cooperative-cost-weights", nargs="+", type=float, metavar="W",
        default=None,
        help="convex player-cost weights for weighted_sum; must be nonnegative and sum to 1",
    )
    parser.add_argument(
        "--disagreement-costs", nargs="+", type=float, metavar="B",
        help=(
            "fixed costs-to-go override; by default each later iteration uses "
            "the previous totals minus costs executed so far"
        ),
    )
    parser.add_argument(
        "--movie", default="LDG_Simulation.mp4", metavar="PATH",
        help="save the evolving figure as an MP4 or GIF (default: LDG_Simulation.mp4)",
    )
    parser.add_argument(
        "--movie-fps", type=float, default=5.0,
        help="movie frames per second (default: 10)",
    )
    parser.add_argument(
        "--movie-dpi", type=float, default=100.0,
        help="movie resolution in dots per inch (default: 100)",
    )
    parser.add_argument(
        "--no-movie", action="store_true",
        help="disable movie export",
    )
    args = parser.parse_args()
    cooperative = cooperative_mode or args.cooperative
    if args.players == 3:
        gamma1_grid = np.asarray(
            args.bargaining_gamma1
            if args.bargaining_gamma1 is not None else bargaining_gamma1,
            dtype=float,
        )
        gamma2_grid = np.asarray(
            args.bargaining_gamma2
            if args.bargaining_gamma2 is not None else bargaining_gamma2,
            dtype=float,
        )
        if gamma1_grid.shape != gamma2_grid.shape:
            raise ValueError("bargaining_gamma1 and bargaining_gamma2 must have equal lengths")
        gamma_grid = np.column_stack((gamma1_grid, gamma2_grid))
        if np.any(gamma_grid < 0.0) or np.any(np.sum(gamma_grid, axis=1) > 1.0 + 1e-12):
            raise ValueError("every bargaining pair must satisfy gamma1 + gamma2 <= 1")
    else:
        gamma_grid = (
            np.asarray(args.bargaining_gammas, dtype=float)
            if args.bargaining_gammas is not None else bargaining_gammas
        )
    baseline_costs = (
        tuple(args.disagreement_costs)
        if args.disagreement_costs is not None else disagreement_costs
    )
    selection_method = args.cooperative_selection
    cost_weights = np.asarray(
        args.cooperative_cost_weights
        if args.cooperative_cost_weights is not None
        else np.full(args.players, 1.0 / args.players),
        dtype=float,
    )
    player_count = args.players
    if player_count == 3 and not cooperative:
        raise ValueError("the three-player simulation requires cooperative control")
    targets = [x1f, x2f] + ([x3f] if player_count == 3 else [])
    x0 = np.concatenate(x0_players[:player_count])
    
    Game = GameDynamics(
        dt, x0, x1f, x2f, x3f=x3f if player_count == 3 else None,
        L=L, W=W, dynamics_type=dynamics_type, v_min=v_min, v_max=v_max,
        a_max=a_max,
        psi_dot_max=psi_dot_max, an_max=an_max,
        MaxIterations=Niterations,
    )
    LearnedData = init_learned_data()
    # To reuse saved data instead: LearnedData = load_learned_data(learned_data_path)
    # LearnedData = load_learned_data(learned_data_path)
    
    Solver2 = DGSolver(
        Game, x1f=x1f, x2f=x2f,
        x3f=x3f if player_count == 3 else None,
        alpha=np.array([alpha1, alpha2]) if player_count == 3 else 1.0 - alpha1,
    )
    plot_simulation_init(Game)
    movie_path = None
    if not args.no_movie:
        movie_path = start_simulation_movie(
            args.movie, fps=args.movie_fps, dpi=args.movie_dpi
        )

    # Start Julia/PATHSolver once for this simulation execution. The main
    # process and persistent terminal workers are reused by every iteration.
    initialize_pathsolver_runtime(max_workers=max_workers)

    for iter in range(Game.Max_Iterations):
        Game.reset_game()
        Solver1 = DGSolver(
            Game, x1f=x1f, x2f=x2f, LearnedData=LearnedData,
            x3f=x3f if player_count == 3 else None,
            alpha=(np.array([alpha1, alpha2]) if player_count == 3 else alpha1),
            horizon=10,
            prev_best_cost=prev_p1_total_cost if iter > 0 else np.inf,
            max_workers=max_workers,
            cooperative=cooperative,
            bargaining_gammas=gamma_grid,
            cooperative_selection=selection_method,
            cooperative_cost_weights=cost_weights,
            disagreement_costs=baseline_costs,
        )        
        
        EndGame = False
        current_cost1 = 0.0
        current_cost2 = 0.0
        current_cost3 = 0.0
        shared_constraint_active = False
        while not EndGame:
            previous_iteration_costs = (
                tuple([prev_p1_total_cost, prev_p2_total_cost]
                      + ([prev_p3_total_cost] if player_count == 3 else []))
                if cooperative and iter > 0 else None
            )
            active_disagreement_costs = baseline_costs
            if cooperative and active_disagreement_costs is None and iter > 0:
                active_disagreement_costs = remaining_cost_budget(
                    previous_iteration_costs,
                    tuple([current_cost1, current_cost2]
                          + ([current_cost3] if player_count == 3 else [])),
                )
            if iter == 0:
                u1 = np.concatenate(
                    (Game.SimpleController1(), Game.SimpleController2(),
                     *([Game.SimpleController3()] if player_count == 3 else []))
                )
                Solver2.Solution.success = False
            else:
                if float(ca.bilin(Solver1.Qk, Game.x[:Game.nx1] - Game.x1f)) <= 1e-8:
                    u1 = np.zeros(Game.nu)
                # Player 1 Controller
                elif (getattr(Solver1.Solution, "terminal_sample_state", None) is not None and 
                      float(ca.bilin(Solver1.Qk, Solver1.Solution.terminal_sample_state[:Game.nx1] - Game.x1f)) <= 1e-8 and
                      float(ca.bilin(Solver1.Qk, Game.x[Game.nx1:2 * Game.nx1] - Game.x2f)) <= 1e-8 and
                      (0 if player_count < 3 else float(ca.bilin(Solver1.Qk, Game.x[2*Game.nx1:] - Game.x3f)))  <= 1e-8 and
                      Solver1.Solution.success and iter > 0):
                    Solver1.Solution.indx += 1
                    u1 = np.concatenate([
                        getattr(Solver1.Solution, f"u{player + 1}")[Solver1.Solution.indx]
                        for player in range(player_count)
                    ])
                else:
                    indx = getattr(Solver1.Solution, "indx", 0)
                    if not Solver1.Solution.success and indx >= int(Solver1.N):
                        u1 = Solver1.step(
                            Game.t, Game.x, current_cost1=current_cost1,
                            current_cost2=current_cost2,
                            current_cost3=current_cost3,
                            use_all_terminal_points=True,
                            disagreement_costs=active_disagreement_costs,
                            previous_iteration_costs=previous_iteration_costs,
                        )
                    else:
                        u1 = Solver1.step(
                            Game.t, Game.x, current_cost1=current_cost1,
                            current_cost2=current_cost2,
                            current_cost3=current_cost3,
                            disagreement_costs=active_disagreement_costs,
                            previous_iteration_costs=previous_iteration_costs,
                        )
                        indx = getattr(Solver1.Solution, "indx", 0)
                        if indx > 0:
                            Found = False
                            u1_N = np.zeros(Game.nu)
                            for dN in [1, 2, 3]:
                                Solver1_N = DGSolver(
                                    Game, x1f=x1f, x2f=x2f,
                                    x3f=x3f if player_count == 3 else None,
                                    LearnedData=LearnedData,
                                    alpha=(np.array([alpha1, alpha2]) if player_count == 3 else alpha1),
                                    max_workers=max_workers,
                                    prev_best_cost=(
                                        prev_p1_total_cost if iter > 0 else np.inf
                                    ),
                                    horizon=Solver1.N+dN,
                                    cooperative=cooperative,
                                    bargaining_gammas=gamma_grid,
                                    cooperative_selection=selection_method,
                                    cooperative_cost_weights=cost_weights,
                                    disagreement_costs=baseline_costs,
                                )
                                u1_N = Solver1_N.step(
                                    Game.t, Game.x, current_cost1=current_cost1,
                                    current_cost2=current_cost2,
                                    current_cost3=current_cost3,
                                    disagreement_costs=active_disagreement_costs,
                                    previous_iteration_costs=previous_iteration_costs,
                                )
                                if Solver1_N.Solution.success:
                                    Found = True
                                    Solver1.Solution = copy.deepcopy(Solver1_N.Solution)
                                    u1 = u1_N
                                    break
                        
                                
            # # Player 2 Controller
            if iter == 0:
                u2 = u1
            elif cooperative:
                u2 = u1
                Solver2.Solution = copy.deepcopy(Solver1.Solution)
            else:
                Solver2.Solution.success = False
                if float(ca.bilin(Solver2.Qk, Game.x[Game.nx1:] - Game.x2f)) <= 1e-8:
                    u2 = np.zeros(Game.nu)
                elif Solver1.Solution.success and iter > 0 and np.size(Solver1.Solution.u1) == Solver1.N and np.size(Solver1.Solution.u2) == Solver1.N:
                    u2 = Solver2.step(Game.t, Game.x, u1_0=Solver1.Solution.u1, u2_0=Solver1.Solution.u2)
                if not Solver2.Solution.success:
                    u2 = Solver2.step(Game.t, Game.x)
                if not Solver2.Solution.success and iter > 0:
                    u1_0 = Solver1.Solution.u1; u1_0[:-1] = u1_0[1:]
                    u2_0 = Solver1.Solution.u2; u2_0[:-1] = u2_0[1:]
                    u2 = Solver2.step(Game.t, Game.x, u1_0=u1_0, u2_0=u2_0, last_attempted_solution=True)

            # calculate current cost for player 1:
            current_cost1 += float(Solver1.stage_costs[0](
                Game.x[:Game.nx1], u1[:Game.nu1]))
            current_cost2 += float(Solver1.stage_costs[1](
                Game.x[Game.nx1:2 * Game.nx1],
                u2[Game.nu1:2 * Game.nu1]))
            if player_count == 3:
                current_cost3 += float(Solver1.stage_costs[2](
                    Game.x[2 * Game.nx1:3 * Game.nx1],
                    u1[2 * Game.nu1:3 * Game.nu1]))
            
            if not cooperative:
                u = np.concatenate((u1[0:2], u2[2:]))
            else:
                u = u1.copy()
            shared_constraint_active |= is_shared_constraint_active(Game, Game.x, u)
            GameFlag = Game.step(u=u)
            shared_constraint_active |= is_shared_constraint_active(Game, Game.x, u)
            plot_simulation(Game, Solver1, Solver2, LearnedData)
            
            default_gamma = (
                np.array([alpha1, alpha2]) if player_count == 3 else alpha1
            )
            selected_gamma = getattr(
                Solver1.Solution, "bargaining_gamma", default_gamma
            )
            if selected_gamma is None:
                selected_gamma = default_gamma
            record_learned_state(LearnedData, Game, iter, selected_gamma)
            if GameFlag != Game.STEP_OK:
                print("Infeasible Step - Stopping Iteration")

            player1_distance = float(ca.bilin(Solver1.Qk, Game.x[:Game.nx1] - Game.x1f))
            player2_distance = float(ca.bilin(
                Solver2.Qk,
                Game.x[Game.nx1:2 * Game.nx1] - Game.targets[1],
            ))
            player_distances = [player1_distance, player2_distance]
            if player_count == 3:
                player3_distance = float(ca.bilin(
                    Solver1.Qk,
                    Game.x[2 * Game.nx1:3 * Game.nx1] - Game.targets[2],
                ))
                player_distances.append(player3_distance)
            
            # if player1_distance <= 10*Solver1.proximity_minval:
            #     Game.x[:Game.nx1] = Game.x1f.copy()
            # if player2_distance <= 10*Solver2.proximity_minval:
            #     Game.x[Game.nx1:] = Game.x2f.copy()

            if Game.t >= tf: EndGame = True
            if GameFlag is not Game.STEP_OK: EndGame = True
            if max(player_distances) <= Solver1.proximity_minval: EndGame = True
            
            print( f"Time: {Game.t:2.2}, "
                   f"Player 1 Dist: {player1_distance:2.2}, "
                   f"Player 2 Dist: {player2_distance:2.2}"
                   + (f", Player 3 Dist: {player3_distance:2.2}" if player_count == 3 else "") )
        
        (LearnedData.RawData[iter].p1_arrival_time, LearnedData.RawData[iter].p2_arrival_time) = arrival_times(Game.get_history(), 0.0, x1f, x2f, Game.nx1, arrival_tolerance,)
        if (np.isfinite(LearnedData.RawData[iter].p1_arrival_time)
            and np.isfinite(LearnedData.RawData[iter].p2_arrival_time)):
            LearnedData.RawData[iter].arrival_time_difference = (
                LearnedData.RawData[iter].p1_arrival_time
                - LearnedData.RawData[iter].p2_arrival_time)
        else:
            LearnedData.RawData[iter].arrival_time_difference = np.nan

        for player in range(player_count):
            start_x = player * Game.nx1
            current_terminal_cost = float(Solver1.stage_costs[player](
                Game.x[start_x:start_x + Game.nx1], np.zeros(Game.nu1)
            ))
            if player == 0:
                current_cost1 += current_terminal_cost
            elif player == 1:
                current_cost2 += current_terminal_cost
            else:
                current_cost3 += current_terminal_cost
        if GameFlag is Game.STEP_OK:
            append_terminal_learned_state(LearnedData, Game, iter)
        
        if EndGame and GameFlag is not Game.STEP_OK:
            exception_message = (
                f"Game ended with an infeasible step at time {Game.t:2.2f} "
                f"and iteration {iter}."
            )
            break

        # if iter > 1:
        #     alpha1 = max(0.0, alpha1-0.05)

        rebuild_analyzed_data(
            LearnedData,
            iter,
            Game,
            Solver1,
            iterations_to_use = max(4, int(max_workers/6)))

        LearnedData.RawData[iter].shared_constraint_active = shared_constraint_active
        if not cooperative and iter > 0 and should_reduce_alpha(
            LearnedData.RawData[iter - 1].p1_total_cost,
            LearnedData.RawData[iter].p1_total_cost,
            shared_constraint_active,
            max_relative_drop = 0.01
        ):
            alpha1 = max(0.0, alpha1 - 0.05)
            print(f"Reduced alpha1 to {alpha1:.2f}")

        if iter > 0:
            costs_not_improving = (
                LearnedData.RawData[-1].p1_total_cost >= LearnedData.RawData[-2].p1_total_cost - 1e-3
                and LearnedData.RawData[-1].p2_total_cost >= LearnedData.RawData[-2].p2_total_cost - 1e-3
            )
            if player_count == 3:
                costs_not_improving &= (
                    LearnedData.RawData[-1].p3_total_cost
                    >= LearnedData.RawData[-2].p3_total_cost - 1e-3
                )
            if costs_not_improving:
                break
        
        prev_p1_total_cost = LearnedData.RawData[iter].p1_total_cost
        prev_p2_total_cost = LearnedData.RawData[iter].p2_total_cost
        if player_count == 3:
            prev_p3_total_cost = LearnedData.RawData[iter].p3_total_cost
        Solver2.Solution.success = False

    save_learned_data(LearnedData, learned_data_path)
    plot_simulation(Game, Solver1, Solver2, LearnedData, pause=None)
    figure_path = save_simulation_figure()
    movie_path = finish_simulation_movie()
    close_simulation_plots()
    print(f"Saved figure to {figure_path}")
    if movie_path is not None:
        print(f"Saved movie to {movie_path}")
    print(f"Saved learned data to {learned_data_path}")
    print("Done!!!")
