import pickle
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np


def save_learned_data(learned_data, path="LearnedData.pkl"):
    """Serialize learned simulation data so it can be reused later."""
    path = Path(path)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("wb") as file:
        pickle.dump(learned_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)
    return path


def load_learned_data(path="LearnedData.pkl"):
    """Load learned simulation data from a trusted pickle file."""
    path = Path(path)
    with path.open("rb") as file:
        learned_data = pickle.load(file)

    if not hasattr(learned_data, "RawData") or not hasattr(
        learned_data, "AnalyzedData"
    ):
        raise ValueError(f"{path} does not contain valid learned data")
    return learned_data


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
    analyzed_data.u2 = []
    analyzed_data.Cost2Go = []
    analyzed_data.n_data = 0
    return analyzed_data


def arrival_times(history, start_time, x1f, x2f, nx1, tolerance):
    """Return P1 and P2 arrival times from start_time onward."""
    times = history["t"]
    states = history["x"]
    target1_position = np.asarray(x1f, dtype=float).reshape(-1)[:2]
    target2_position = np.asarray(x2f, dtype=float).reshape(-1)[:2]
    future = times >= start_time

    player_arrival_times = []
    for position_indices, target_position in zip(([0, 1], [nx1, nx1 + 1]), (target1_position, target2_position)):
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
    iterations_to_use=5,
):
    """Rebuild analyzed data using only the latest RawData iterations."""
    analyzed_data = init_analyzed_data()
    first_iteration = max(0, current_iteration - iterations_to_use + 1)
    stop_iteration = first_iteration - 1 if first_iteration > 0 else None

    for raw_data in learned_data.RawData[current_iteration + 1:stop_iteration:-1]:
        states = raw_data.x
        p1_stage_costs = [(solver.l1(state[:game.nx1], u[:game.nu1])) for state, u in zip(states, raw_data.u)]
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

            # analyzed_data.c.append(np.array([cx_min, cx_max, cy_min, cy_max]))
            analyzed_data.Cost2Go.append(p1_cost_to_go)
            analyzed_data.u2.append(u[2:4])

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


def append_terminal_learned_state(learned_data, game, iteration):
    """Append a zero-cost target sample one time step after the simulation."""
    target1_state = np.asarray(game.x1f, dtype=float).reshape(-1)
    target2_state = np.asarray(game.x2f, dtype=float).reshape(-1)
    if target1_state.shape != (game.nx1,):
        raise ValueError(
            f"x1f must contain one player state with shape ({game.nx1},)"
        )
    if target2_state.shape != (game.nx2,):
        raise ValueError(
            f"x2f must contain one player state with shape ({game.nx2},)"
        )

    raw_data = learned_data.RawData[iteration]
    raw_data.t.append(float(game.t + game.dt))
    raw_data.x.append(np.concatenate((target1_state, target2_state)))
    raw_data.u.append(np.zeros(game.nu, dtype=float))
