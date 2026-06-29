import os
os.system('clear')

import numpy as np
import casadi as ca
import copy

from Game import GameDynamics

from DGSolver import DGSolver, initialize_pathsolver_runtime
from SimulationPlot import *
from types import SimpleNamespace

np.random.seed(100)

def player_state(px, py, vx=0.0, vy=0.0):
    if dynamics_type == 1:
        x = [px + 0.0*np.random.normal(), py + 0.0*np.random.normal()]
        return x
    return [px, py, vx, vy]

L = 5.0
W = 4.0
dt = 0.1
tf = 7.0
dynamics_type = 1  # 1: single integrator, 2: double integrator
terminal_constraint_mode = "sampled_points" # {"convex_hull", "sampled_points"}
Niterations = 10
arrival_tolerance = 0.1
xf = np.array([player_state(1.0, 1.5)])

def init_learned_data():
    LearnedData = SimpleNamespace()
    LearnedData.RawData = []
    LearnedData.AnalyzedData = init_analyzed_data()
    return LearnedData
 

def init_analyzed_data():
    analyzed_data = SimpleNamespace()
    analyzed_data.t = []
    analyzed_data.state = []
    analyzed_data.c = []
    analyzed_data.Cost2Go = []
    analyzed_data.n_data = 0
    return analyzed_data


def arrival_times(history, start_time, xf, nx1, tolerance):
    """Return P1 and P2 arrival times from start_time onward."""
    times = history["t"]
    states = history["x"]
    target_position = np.asarray(xf, dtype=float).reshape(-1)[:2]
    future = times >= start_time

    arrival_times = []
    for position_indices in ([0, 1], [nx1, nx1 + 1]):
        distance = np.linalg.norm(states[:, position_indices] - target_position, axis=1)
        arrivals = np.flatnonzero(future & (distance <= tolerance))
        if arrivals.size == 0:
            arrival_times.append(np.nan)
            continue
        arrival_times.append(times[arrivals[0]] - start_time)

    return tuple(float(arrival_time) for arrival_time in arrival_times)


def arrival_time_difference(history, start_time, xf, nx1, tolerance):
    """Return P1 arrival time minus P2 arrival time from start_time onward."""
    p1_arrival_time, p2_arrival_time = arrival_times(
        history,
        start_time,
        xf,
        nx1,
        tolerance,
    )
    if not (np.isfinite(p1_arrival_time) and np.isfinite(p2_arrival_time)):
        return np.nan
    return float(p1_arrival_time - p2_arrival_time)


def rebuild_analyzed_data(
    learned_data,
    current_iteration,
    game,
    solver,
    xf,
    tolerance,
    iterations_to_use=2,
):
    """Rebuild analyzed data using only the latest RawData iterations."""
    analyzed_data = init_analyzed_data()
    first_iteration = max(0, current_iteration - iterations_to_use + 1)

    for raw_data in learned_data.RawData[first_iteration:current_iteration + 1]:
        states = raw_data.x
        p1_stage_costs = [
            (
                np.array(ca.bilin(solver.Qk, state[:game.nx1] - xf))[0, 0]
                + np.array(ca.bilin(solver.R1 * np.eye(game.nu1), u[:game.nu1]))[0, 0]
            )
            for state, u in zip(states, raw_data.u)
        ]
        p1_costs_to_go = np.cumsum(p1_stage_costs[::-1])[::-1]
        raw_data.p1_total_cost = float(p1_costs_to_go[0])

        for t, state, u, p1_cost_to_go in zip(
            raw_data.t,
            states,
            raw_data.u,
            p1_costs_to_go,
        ):
            if any(
                ca.bilin(solver.proximity_Q, saved_state - state)
                < solver.proximity_minval
                for saved_state in analyzed_data.state
            ):
                continue
            
            analyzed_data.t.append(t)
            analyzed_data.state.append(state)

            cx_max = game.u_max_shared - u[2]
            cx_min = game.u_min_shared - u[2]
            cy_max = game.u_max_shared - u[3]
            cy_min = game.u_min_shared - u[3]
            analyzed_data.c.append(
                np.array([cx_min, cx_max, cy_min, cy_max])
            )
            # analyzed_data.Cost2Go.append(
            #     arrival_time_difference(
            #         history,
            #         t,
            #         xf,
            #         game.nx1,
            #         tolerance,
            #     ) )
            analyzed_data.Cost2Go.append(p1_cost_to_go)
                
           
    analyzed_data.n_data = len(analyzed_data.state)
    learned_data.AnalyzedData = analyzed_data


def record_learned_state(learned_data, game, iter, alpha, feasible = True):
    if len(learned_data.RawData) < iter+1:
        learned_data.RawData.append(SimpleNamespace())
        learned_data.RawData[iter].alpha = float(alpha)
        learned_data.RawData[iter].t = []
        learned_data.RawData[iter].x = []
        learned_data.RawData[iter].u = []
        learned_data.RawData[iter].arrival_time_difference = np.nan
        learned_data.RawData[iter].p1_arrival_time = np.nan
        learned_data.RawData[iter].p2_arrival_time = np.nan
        
    if feasible:
        learned_data.RawData[iter].t.append(float(game.history['t'][-2]))
        learned_data.RawData[iter].x.append(game.history['x'][-2].copy())
        learned_data.RawData[iter].u.append(game.history['u'][-1].copy())
    else:
        learned_data.RawData[iter].A = []
        learned_data.RawData[iter].b = []


def append_terminal_learned_state(learned_data, game, iter, xf):
    """Append a zero-cost target sample one time step after the simulation."""
    target_state = np.asarray(xf, dtype=float).reshape(-1)
    if target_state.shape != (game.nx1,):
        raise ValueError(f"xf must contain one player state with shape ({game.nx1},)")

    raw_data = learned_data.RawData[iter]
    raw_data.t.append(float(game.t + game.dt))
    raw_data.x.append(np.concatenate((target_state, target_state)))
    raw_data.u.append(np.zeros(game.nu, dtype=float))
        

if __name__ == '__main__':
    
    x0 = np.array(player_state(0.5-L/2, 0.5) + player_state(0.5-L/2, -1.5))
    alpha1, alpha2 = 1.0, 0.49
    
    Game = GameDynamics(dt, x0, xf, L=L, W=W, dynamics_type=dynamics_type)
    LearnedData = init_learned_data()
    
    Solver2 = DGSolver(Game, xf=xf, alpha=alpha2)
    plot_simulation_init(Game)

    # Start Julia/PATHSolver once for this simulation execution. The main
    # process and persistent terminal workers are reused by every iteration.
    initialize_pathsolver_runtime()

    for iter in range(Niterations):
        Game.reset_game()
        Solver1 = DGSolver(Game, xf=xf, LearnedData=LearnedData, alpha=alpha1)
        Solver1.alpha_vec[0] = 1.0
        EndGame = False
        while not EndGame:
            # Player 1 Controller
            u1 = Solver1.step(Game.t, Game.x)

            if not Solver1.Solution.success:
                u1_0 = Solver1.Solution.u1; u1_0[:-1] = u1_0[1:]
                u2_0 = Solver1.Solution.u2; u2_0[:-1] = u2_0[1:]
                u1 = Solver1.step(Game.t, Game.x, u1_0=u1_0, u2_0=u2_0)
            if not Solver1.Solution.success:
                for alpha in [1.0, 0.9, 0.8, 0.7]:
                    try:
                        u1 = Solver1.step(Game.t, Game.x, forced_alpha=alpha)
                        if Solver1.Solution.success: break
                    except:
                        pass
                                
            # # Player 2 Controller
            if Solver1.Solution.success:
                u2 = Solver2.step(Game.t, Game.x, u1_0=Solver1.Solution.u1, u2_0=Solver1.Solution.u2)
            if not Solver2.Solution.success:
                u2 = Solver2.step(Game.t, Game.x)
            if not Solver2.Solution.success:
                u1_0 = Solver1.Solution.u1; u1_0[:-1] = u1_0[1:]
                u2_0 = Solver1.Solution.u2; u2_0[:-1] = u2_0[1:]
                u2 = Solver2.step(Game.t, Game.x, u1_0=u1_0, u2_0=u2_0)
        
            u = np.concatenate((u1[0:2], u2[2:]))
            GameFlag = Game.step(u=u)
            plot_simulation(Game, Solver1, Solver2)
            
            record_learned_state(LearnedData, Game, iter, alpha1)
            if GameFlag != Game.STEP_OK:
                print("Infeasible Step - Stopping Iteration")

            target_position = np.asarray(xf, dtype=float).reshape(-1)[:2]
            player1_distance = np.linalg.norm(Game.x[:2] - target_position)
            player2_distance = np.linalg.norm(
                Game.x[Game.nx1:Game.nx1 + 2] - target_position
            )

            if Game.t >= tf: EndGame = True
            if GameFlag is not Game.STEP_OK: EndGame = True
            if (
                player1_distance <= arrival_tolerance/2
                and player2_distance <= arrival_tolerance/2
            ):
                EndGame = True
            
            print( f"Time:{Game.t:2.2}, "
                   f"Player 1 Dist: {player1_distance:2.2}, "
                   f"Player 2 Dist: {player2_distance:2.2}" )
        
        (
            LearnedData.RawData[iter].p1_arrival_time,
            LearnedData.RawData[iter].p2_arrival_time,
        ) = arrival_times(
            Game.get_history(),
            0.0,
            xf,
            Game.nx1,
            arrival_tolerance,
        )
        if (
            np.isfinite(LearnedData.RawData[iter].p1_arrival_time)
            and np.isfinite(LearnedData.RawData[iter].p2_arrival_time)
        ):
            LearnedData.RawData[iter].arrival_time_difference = (
                LearnedData.RawData[iter].p1_arrival_time
                - LearnedData.RawData[iter].p2_arrival_time
            )
        else:
            LearnedData.RawData[iter].arrival_time_difference = np.nan

        append_terminal_learned_state(LearnedData, Game, iter, xf)

        rebuild_analyzed_data(
            LearnedData,
            iter,
            Game,
            Solver1,
            xf,
            arrival_tolerance,
        )
        
        Solver2.Solution.success = False

    plot_simulation(Game, Solver1, Solver2, pause=None)
    figure_path = save_simulation_figure()
    close_simulation_plots()
    print(f"Saved figure to {figure_path}")
    print("Done!!!")
