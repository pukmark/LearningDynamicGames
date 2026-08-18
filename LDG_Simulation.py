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
tf = 10.0
dynamics_type = 2  # 1: single integrator, 2: double integrator
terminal_constraint_mode = "sampled_points" # {"convex_hull", "sampled_points"}
# In cooperative mode Solver1 selects both the learned safe-set reconnection
# state and the shared-constraint equilibrium weight by Nash bargaining. Its
# joint control output is applied to both players.
cooperative_mode = True
bargaining_gammas = np.array([0.35, 0.45, 0.5, 0.55, 0.65])
# bargaining_gammas = np.array([0.5])
# Optional fixed (b1_t, b2_t) costs-to-go. When this is None, iterations after
# the bootstrap use the previous completed totals minus costs executed so far.
disagreement_costs = None
Niterations = 10
arrival_tolerance = 0.01
learned_data_path = "LearnedData.pkl"
x1f = np.array([player_state(1.5, -1.5, dynamics_type=dynamics_type)])
x2f = np.array([player_state(-1.5, 1.5, dynamics_type=dynamics_type)])
max_workers = max(1, int(os.cpu_count() * 0.4))
# max_workers = 1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the learning dynamic-game simulation.")
    parser.add_argument(
        "--cooperative", action="store_true",
        help="apply Solver1 to both players and select (safe state, gamma) by Nash bargaining",
    )
    parser.add_argument(
        "--bargaining-gammas", nargs="+", type=float, default=None,
        help="candidate equilibrium weights in [0, 1] (default: 0.1 ... 0.9)",
    )
    parser.add_argument(
        "--disagreement-costs", nargs=2, type=float, metavar=("B1", "B2"),
        help=(
            "fixed costs-to-go override; by default each later iteration uses "
            "the previous totals minus costs executed so far"
        ),
    )
    args = parser.parse_args()
    cooperative = cooperative_mode or args.cooperative
    gamma_grid = (
        np.asarray(args.bargaining_gammas, dtype=float)
        if args.bargaining_gammas is not None else bargaining_gammas
    )
    baseline_costs = (
        tuple(args.disagreement_costs)
        if args.disagreement_costs is not None else disagreement_costs
    )
    
    x0 = np.array( player_state(-1.75, 1.5, dynamics_type=dynamics_type) + player_state(0.0, -1.0, dynamics_type=dynamics_type))
    alpha1, alpha2 = 0.5, 0.5
    
    Game = GameDynamics(dt, x0, x1f, x2f, L=L, W=W, dynamics_type=dynamics_type, MaxIterations=Niterations)
    LearnedData = init_learned_data()
    # To reuse saved data instead: LearnedData = load_learned_data(learned_data_path)
    # LearnedData = load_learned_data(learned_data_path)
    
    Solver2 = DGSolver(Game, x1f=x1f, x2f=x2f, alpha=alpha2)
    plot_simulation_init(Game)

    # Start Julia/PATHSolver once for this simulation execution. The main
    # process and persistent terminal workers are reused by every iteration.
    initialize_pathsolver_runtime(max_workers=max_workers)

    for iter in range(Game.Max_Iterations):
        Game.reset_game()
        Solver1 = DGSolver(
            Game, x1f=x1f, x2f=x2f, LearnedData=LearnedData,
            alpha=alpha1, max_workers=max_workers,
            prev_best_cost=prev_p1_total_cost if iter > 0 else np.inf,
            cooperative=cooperative,
            bargaining_gammas=gamma_grid,
            disagreement_costs=baseline_costs,
        )
        # Solver1 = DGSolver(Game, x1f=x1f, x2f=x2f, LearnedData=LearnedData, alpha=alpha1, max_workers=max_workers, prev_best_cost=np.inf)
        EndGame = False
        current_cost1 = 0.0
        current_cost2 = 0.0
        shared_constraint_active = False
        while not EndGame:
            previous_iteration_costs = (
                (prev_p1_total_cost, prev_p2_total_cost)
                if cooperative and iter > 0 else None
            )
            active_disagreement_costs = baseline_costs
            if cooperative and active_disagreement_costs is None and iter > 0:
                active_disagreement_costs = remaining_cost_budget(
                    (prev_p1_total_cost, prev_p2_total_cost),
                    (current_cost1, current_cost2),
                )
            if iter == 0:
                u1 = np.concatenate(
                    (Game.SimpleController1(), Game.SimpleController2())
                )
                Solver2.Solution.success = False
            else:
                if float(ca.bilin(Solver1.Qk, Game.x[:Game.nx1] - Game.x1f)) <= 1e-8:
                    u1 = np.zeros(Game.nu)
                # Player 1 Controller
                elif getattr(Solver1.Solution, "terminal_sample_state", None) is not None and float(ca.bilin(Solver1.Qk, Solver1.Solution.terminal_sample_state[:Game.nx1] - Game.x1f)) <= 1e-8:
                    Solver1.Solution.indx += 1
                    u1 = np.concatenate( (Solver1.Solution.u1[Solver1.Solution.indx],Solver1.Solution.u2[Solver1.Solution.indx]))
                else:
                    indx = getattr(Solver1.Solution, "indx", 0)
                    if not Solver1.Solution.success and indx >= int(Solver1.N):
                        u1 = Solver1.step(
                            Game.t, Game.x, current_cost1=current_cost1,
                            current_cost2=current_cost2,
                            use_all_terminal_points=True,
                            disagreement_costs=active_disagreement_costs,
                            previous_iteration_costs=previous_iteration_costs,
                        )
                    else:
                        u1 = Solver1.step(
                            Game.t, Game.x, current_cost1=current_cost1,
                            current_cost2=current_cost2,
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
                                    LearnedData=LearnedData, alpha=alpha1,
                                    max_workers=max_workers,
                                    prev_best_cost=(
                                        prev_p1_total_cost if iter > 0 else np.inf
                                    ),
                                    horizon=Solver1.N+dN,
                                    cooperative=cooperative,
                                    bargaining_gammas=gamma_grid,
                                    disagreement_costs=baseline_costs,
                                )
                                u1_N = Solver1_N.step(
                                    Game.t, Game.x, current_cost1=current_cost1,
                                    current_cost2=current_cost2,
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
            current_cost1 += float(Solver1.l1(Game.x[:Game.nx1], u1[:Game.nu1], Game.x[Game.nx1:], u2[Game.nu1:]))
            current_cost2 += float(Solver1.l2(Game.x[Game.nx1:], u2[Game.nu1:], Game.x[:Game.nx1], u1[:Game.nu1]))
            
            if not cooperative:
                u = np.concatenate((u1[0:2], u2[2:]))
            else:
                u = np.concatenate((u1[0:2], u1[2:]))
            shared_constraint_active |= is_shared_constraint_active(Game, Game.x, u)
            GameFlag = Game.step(u=u)
            shared_constraint_active |= is_shared_constraint_active(Game, Game.x, u)
            plot_simulation(Game, Solver1, Solver2, LearnedData)
            
            selected_gamma = getattr(Solver1.Solution, "bargaining_gamma", alpha1)
            if selected_gamma is None:
                selected_gamma = alpha1
            record_learned_state(LearnedData, Game, iter, selected_gamma)
            if GameFlag != Game.STEP_OK:
                print("Infeasible Step - Stopping Iteration")

            player1_distance = float(ca.bilin(Solver1.Qk, Game.x[:Game.nx1] - Game.x1f))
            player2_distance = float(ca.bilin(Solver2.Qk, Game.x[Game.nx1:] - Game.x2f))
            
            # if player1_distance <= 10*Solver1.proximity_minval:
            #     Game.x[:Game.nx1] = Game.x1f.copy()
            # if player2_distance <= 10*Solver2.proximity_minval:
            #     Game.x[Game.nx1:] = Game.x2f.copy()

            if Game.t >= tf: EndGame = True
            if GameFlag is not Game.STEP_OK: EndGame = True
            if ( max(player1_distance, player2_distance) <= Solver1.proximity_minval): EndGame = True
            
            print( f"Time: {Game.t:2.2}, "
                   f"Player 1 Dist: {player1_distance:2.2}, "
                   f"Player 2 Dist: {player2_distance:2.2}" )
        
        (LearnedData.RawData[iter].p1_arrival_time, LearnedData.RawData[iter].p2_arrival_time) = arrival_times(Game.get_history(), 0.0, x1f, x2f, Game.nx1, arrival_tolerance,)
        if (np.isfinite(LearnedData.RawData[iter].p1_arrival_time)
            and np.isfinite(LearnedData.RawData[iter].p2_arrival_time)):
            LearnedData.RawData[iter].arrival_time_difference = (
                LearnedData.RawData[iter].p1_arrival_time
                - LearnedData.RawData[iter].p2_arrival_time)
        else:
            LearnedData.RawData[iter].arrival_time_difference = np.nan

        current_cost1 += float(Solver1.l1(Game.x[:Game.nx1], np.zeros_like(u1[:Game.nu1]), Game.x[Game.nx1:], np.zeros_like(u2[Game.nu1:])))
        current_cost2 += float(Solver1.l2(Game.x[Game.nx1:], np.zeros_like(u2[Game.nu1:]), Game.x[:Game.nx1], np.zeros_like(u1[:Game.nu1])))
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
        
        prev_p1_total_cost = LearnedData.RawData[iter].p1_total_cost
        prev_p2_total_cost = LearnedData.RawData[iter].p2_total_cost
        Solver2.Solution.success = False

    save_learned_data(LearnedData, learned_data_path)
    plot_simulation(Game, Solver1, Solver2, LearnedData, pause=None)
    figure_path = save_simulation_figure()
    close_simulation_plots()
    print(f"Saved figure to {figure_path}")
    print(f"Saved learned data to {learned_data_path}")
    print("Done!!!")
