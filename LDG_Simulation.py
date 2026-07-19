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
    load_learned_data,
    player_state,
    rebuild_analyzed_data,
    record_learned_state,
    save_learned_data,
)
from SimulationPlot import *

np.random.seed(100)

L = 5.0
W = 4.0
dt = 0.1
tf = 10.0
dynamics_type = 2  # 1: single integrator, 2: double integrator
terminal_constraint_mode = "sampled_points" # {"convex_hull", "sampled_points"}
Niterations = 40
arrival_tolerance = 0.05
learned_data_path = "LearnedData.pkl"
x1f = np.array([player_state(1.0, 1.5, dynamics_type=dynamics_type)])
x2f = np.array([player_state(-2.0, 1.5, dynamics_type=dynamics_type)])
max_workers = max(1, int(os.cpu_count() * 0.3))
# max_workers = 1
        

if __name__ == '__main__':
    
    x0 = np.array(
        player_state(0.5-L/2, 0.5, dynamics_type=dynamics_type)
        + player_state(L/2-0.5, -1., dynamics_type=dynamics_type)
    )
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
        while not EndGame:
            if iter == 0:
                u1 = Game.SimpleController()
                Solver2.Solution.success = False
            else:
                # Player 1 Controller
                u1 = Solver1.step(Game.t, Game.x)
                
                if not Solver1.Solution.success and Solver1.Solution.indx > int(0.5 * Solver1.N):
                    u1 = Solver1.step(Game.t, Game.x, use_all_terminal_points=True)

                    if not Solver1.Solution.success:
                        u1_0 = Solver1.Solution.u1; u1_0[:-1] = u1_0[1:]
                        u2_0 = Solver1.Solution.u2; u2_0[:-1] = u2_0[1:]
                        u1 = Solver1.step(Game.t, Game.x, u1_0=u1_0, u2_0=u2_0)
                    if not Solver1.Solution.success:
                        for alpha in [0.9, 0.8, 0.7]:
                            try:
                                u1 = Solver1.step(Game.t, Game.x, forced_alpha=alpha)
                                if Solver1.Solution.success: break
                            except:
                                pass
                                
            # # Player 2 Controller
            Solver2.Solution.success = False
            if Solver1.Solution.success and iter > 0:
                u2 = Solver2.step(Game.t, Game.x, u1_0=Solver1.Solution.u1, u2_0=Solver1.Solution.u2)
            if not Solver2.Solution.success:
                u2 = Solver2.step(Game.t, Game.x)
            if not Solver2.Solution.success and iter > 0:
                u1_0 = Solver1.Solution.u1; u1_0[:-1] = u1_0[1:]
                u2_0 = Solver1.Solution.u2; u2_0[:-1] = u2_0[1:]
                u2 = Solver2.step(Game.t, Game.x, u1_0=u1_0, u2_0=u2_0, last_attempted_solution=True)
        
            u = np.concatenate((u1[0:2], u2[2:]))
            GameFlag = Game.step(u=u)
            plot_simulation(Game, Solver1, Solver2, LearnedData)
            
            record_learned_state(LearnedData, Game, iter, alpha1)
            if GameFlag != Game.STEP_OK:
                print("Infeasible Step - Stopping Iteration")

            player1_distance = float(ca.bilin(Solver1.Qk, Game.x[:Game.nx1]            -Game.x1f))
            player2_distance = float(ca.bilin(Solver2.Qk, Game.x[Game.nx1:]-Game.x2f))

            if Game.t >= tf: EndGame = True
            if GameFlag is not Game.STEP_OK: EndGame = True
            if ( player1_distance <= Solver1.proximity_minval and player2_distance <= Solver2.proximity_minval ): EndGame = True
            
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
            raise ValueError(exception_message)

        # if iter > 1:
        #     alpha1 = max(0.0, alpha1-0.05)

        rebuild_analyzed_data(
            LearnedData,
            iter,
            Game,
            Solver1,)
        
        Solver2.Solution.success = False

    save_learned_data(LearnedData, learned_data_path)
    plot_simulation(Game, Solver1, Solver2, LearnedData, pause=None)
    figure_path = save_simulation_figure()
    close_simulation_plots()
    print(f"Saved figure to {figure_path}")
    print(f"Saved learned data to {learned_data_path}")
    print("Done!!!")
