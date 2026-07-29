import os
os.system('clear')
import numpy as np
import casadi as ca

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
Niterations = 15
arrival_tolerance = 0.01
learned_data_path = "LearnedData.pkl"
x1f = np.array([player_state(1.0, -1.5, dynamics_type=dynamics_type)])
x2f = np.array([player_state(-1.0, 1.5, dynamics_type=dynamics_type)])
max_workers = max(1, int(os.cpu_count() * 0.4))
# max_workers = 1
        

if __name__ == '__main__':
    
    x0 = np.array( player_state(0.5-L/2, 1.5, dynamics_type=dynamics_type) + player_state(L/2-0.5, -1.5, dynamics_type=dynamics_type))
    alpha1, alpha2 = 1.0, 0.46
    
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
        Solver1 = DGSolver(Game, x1f=x1f, x2f=x2f, LearnedData=LearnedData, alpha=alpha1, max_workers=max_workers)
        EndGame = False
        current_cost1 = 0.0
        shared_constraint_active = False
        while not EndGame:
            if iter == 0:
                u1 = Game.SimpleController()
                Solver2.Solution.success = False
            else:
                if float(ca.bilin(Solver1.Qk, Game.x[:Game.nx1] - Game.x1f)) <= 1e-8:
                    u1 = np.zeros(Game.nu)
                # Player 1 Controller
                elif Solver1.Solution.terminal_sample_state is not None and float(ca.bilin(Solver1.Qk, Solver1.Solution.terminal_sample_state[:Game.nx1] - Game.x1f)) <= 1e-8:
                    Solver1.Solution.indx += 1
                    u1 = np.concatenate( (Solver1.Solution.u1[Solver1.Solution.indx],Solver1.Solution.u2[Solver1.Solution.indx]))
                else:
                    indx = getattr(Solver1.Solution, "indx", 0)
                    if not Solver1.Solution.success and indx >= int(0.8 * Solver1.N):
                        u1 = Solver1.step(Game.t, Game.x, current_cost1=current_cost1, use_all_terminal_points=True)
                    else:
                        u1 = Solver1.step(Game.t, Game.x, current_cost1=current_cost1)

                if Solver1.Solution.indx >= Solver1.N:
                    Found = False
                    u1 = np.zeros(Game.nu)
                    for dalpha1 in np.linspace(0.0, 0.5, 10)[1:]:
                        u1 = Solver1.step(Game.t, Game.x, current_cost1=current_cost1, use_all_terminal_points=True, forced_alpha=alpha1-dalpha1)
                        if Solver1.Solution.success:
                            Found = True
                            break
                        
                                
            # # Player 2 Controller
            Solver2.Solution.success = False
            if float(ca.bilin(Solver2.Qk, Game.x[Game.nx1:] - Game.x2f)) <= 1e-8:
                u2 = np.zeros(Game.nu)
            elif Solver1.Solution.success and iter > 0:
                u2 = Solver2.step(Game.t, Game.x, u1_0=Solver1.Solution.u1, u2_0=Solver1.Solution.u2)
            if not Solver2.Solution.success:
                u2 = Solver2.step(Game.t, Game.x)
            if not Solver2.Solution.success and iter > 0:
                u1_0 = Solver1.Solution.u1; u1_0[:-1] = u1_0[1:]
                u2_0 = Solver1.Solution.u2; u2_0[:-1] = u2_0[1:]
                u2 = Solver2.step(Game.t, Game.x, u1_0=u1_0, u2_0=u2_0, last_attempted_solution=True)

            # calculate current cost for player 1:
            current_cost1 += float(Solver1.l1(Game.x[:Game.nx1], u1[:Game.nu1]))
            
            u = np.concatenate((u1[0:2], u2[2:]))
            shared_constraint_active |= is_shared_constraint_active(Game, Game.x, u)
            GameFlag = Game.step(u=u)
            shared_constraint_active |= is_shared_constraint_active(Game, Game.x, u)
            plot_simulation(Game, Solver1, Solver2, LearnedData)
            
            record_learned_state(LearnedData, Game, iter, alpha1)
            if GameFlag != Game.STEP_OK:
                print("Infeasible Step - Stopping Iteration")

            player1_distance = float(ca.bilin(Solver1.Qk, Game.x[:Game.nx1] - Game.x1f))
            player2_distance = float(ca.bilin(Solver2.Qk, Game.x[Game.nx1:] - Game.x2f))

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
            Solver1,)

        LearnedData.RawData[iter].shared_constraint_active = shared_constraint_active
        if iter > 0 and should_reduce_alpha(
            LearnedData.RawData[iter - 1].p1_total_cost,
            LearnedData.RawData[iter].p1_total_cost,
            shared_constraint_active,
            max_relative_drop = 0.01
        ):
            alpha1 = max(0.0, alpha1 - 0.05)
            print(f"Reduced alpha1 to {alpha1:.2f}")
        
        Solver2.Solution.success = False

    save_learned_data(LearnedData, learned_data_path)
    plot_simulation(Game, Solver1, Solver2, LearnedData, pause=None)
    figure_path = save_simulation_figure()
    close_simulation_plots()
    print(f"Saved figure to {figure_path}")
    print(f"Saved learned data to {learned_data_path}")
    print("Done!!!")
