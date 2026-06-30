from types import SimpleNamespace

import casadi as ca
import numpy as np


def player_state(px, py, vx=0.0, vy=0.0, dynamics_type=1):
    if dynamics_type == 1:
        return [px + 0.0 * np.random.normal(), py + 0.0 * np.random.normal()]
    return [px, py, vx, vy]


def init_learned_data():
    learned_data = SimpleNamespace()
    learned_data.RawData = []
    learned_data.AnalyzedData = init_analyzed_data()
    return learned_data


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

    player_arrival_times = []
    for position_indices in ([0, 1], [nx1, nx1 + 1]):
        distance = np.linalg.norm(states[:, position_indices] - target_position, axis=1)
        arrivals = np.flatnonzero(future & (distance <= tolerance))
        if arrivals.size == 0:
            player_arrival_times.append(np.nan)
            continue
        player_arrival_times.append(times[arrivals[0]] - start_time)

    return tuple(float(arrival_time) for arrival_time in player_arrival_times)


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
            analyzed_data.c.append(np.array([cx_min, cx_max, cy_min, cy_max]))
            analyzed_data.Cost2Go.append(p1_cost_to_go)

    analyzed_data.n_data = len(analyzed_data.state)
    learned_data.AnalyzedData = analyzed_data


def record_learned_state(learned_data, game, iteration, alpha, feasible=True):
    if len(learned_data.RawData) < iteration + 1:
        learned_data.RawData.append(SimpleNamespace())
        learned_data.RawData[iteration].alpha = float(alpha)
        learned_data.RawData[iteration].t = []
        learned_data.RawData[iteration].x = []
        learned_data.RawData[iteration].u = []
        learned_data.RawData[iteration].arrival_time_difference = np.nan
        learned_data.RawData[iteration].p1_arrival_time = np.nan
        learned_data.RawData[iteration].p2_arrival_time = np.nan

    if feasible:
        learned_data.RawData[iteration].t.append(float(game.history["t"][-2]))
        learned_data.RawData[iteration].x.append(game.history["x"][-2].copy())
        learned_data.RawData[iteration].u.append(game.history["u"][-1].copy())
    else:
        learned_data.RawData[iteration].A = []
        learned_data.RawData[iteration].b = []


def append_terminal_learned_state(learned_data, game, iteration, xf):
    """Append a zero-cost target sample one time step after the simulation."""
    target_state = np.asarray(xf, dtype=float).reshape(-1)
    if target_state.shape != (game.nx1,):
        raise ValueError(
            f"xf must contain one player state with shape ({game.nx1},)"
        )

    raw_data = learned_data.RawData[iteration]
    raw_data.t.append(float(game.t + game.dt))
    raw_data.x.append(np.concatenate((target_state, target_state)))
    raw_data.u.append(np.zeros(game.nu, dtype=float))
