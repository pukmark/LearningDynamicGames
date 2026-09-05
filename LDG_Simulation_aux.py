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


def player_state(px, py, vx=0.5, vy=0.0, dynamics_type=1):
    """Build a player state; vx supplies scalar speed in unicycle mode."""
    if dynamics_type == 1:
        return [px + 0.0 * np.random.normal(), py + 0.0 * np.random.normal()]
    if dynamics_type == 3:
        return [px, py, vx]
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
    analyzed_data.Cost2Go2 = []
    analyzed_data.Cost2Go3 = []
    analyzed_data.n_data = 0
    return analyzed_data


def arrival_times(history, start_time, x1f, x2f, nx1, tolerance):
    """Return when each player enters and subsequently remains near its target."""
    times = history["t"]
    states = history["x"]
    target1_position = np.asarray(x1f, dtype=float).reshape(-1)[:2]
    target2_position = np.asarray(x2f, dtype=float).reshape(-1)[:2]
    future = times >= start_time

    player_arrival_times = []
    for position_indices, target_position in zip(
        ([0, 1], [nx1, nx1 + 1]),
        (target1_position, target2_position),
    ):
        distance = np.linalg.norm(states[:, position_indices] - target_position, axis=1)
        stays_within_tolerance = np.logical_and.accumulate(
            (distance <= tolerance)[::-1]
        )[::-1]
        arrivals = np.flatnonzero(future & stays_within_tolerance)
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


def is_shared_constraint_active(game, state, control, tolerance=None):
    """Return whether any shared-constraint residual is near zero."""
    if tolerance is None:
        tolerance = game.eps

    player_controls = [
        np.asarray(control[p * game.nu1:(p + 1) * game.nu1], dtype=float)
        for p in range(game.n_players)
    ]
    residuals = game.f_shared(np.asarray(state, dtype=float), *player_controls)
    if not isinstance(residuals, tuple):
        residuals = (residuals,)

    return any(
        np.any(np.abs(np.asarray(residual, dtype=float)) <= tolerance)
        for residual in residuals
    )


def should_reduce_alpha(
    previous_cost,
    current_cost,
    shared_constraint_active,
    max_relative_drop=0.01,
):
    """Return whether cost improvement is at most the requested fraction."""
    if shared_constraint_active:
        return False

    previous_cost = float(previous_cost)
    current_cost = float(current_cost)
    if not (np.isfinite(previous_cost) and np.isfinite(current_cost)):
        return False

    cost_drop = previous_cost - current_cost
    return cost_drop <= max_relative_drop * abs(previous_cost)


def remaining_cost_budget(previous_total_costs, executed_costs):
    """Return the costs-to-go that keep both totals within the prior iteration.

    The returned disagreement point is ``previous total - cost already
    executed`` for each player and is therefore intended to be recomputed at
    every receding-horizon step.
    """
    previous = np.asarray(previous_total_costs, dtype=float).reshape(-1)
    executed = np.asarray(executed_costs, dtype=float).reshape(-1)
    if previous.shape != executed.shape or previous.size not in (2, 3):
        raise ValueError(
            "previous_total_costs and executed_costs must contain the same two or three costs"
        )
    if not (np.all(np.isfinite(previous)) and np.all(np.isfinite(executed))):
        raise ValueError("cost budgets require finite previous and executed costs")
    return previous - executed


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
        stage_costs = getattr(solver, "stage_costs", None)
        if stage_costs is None:
            p1_stage_costs = [
                solver.l1(state[:game.nx1], u[:game.nu1],
                          state[game.nx1:], u[game.nu1:])
                for state, u in zip(states, raw_data.u)
            ]
        else:
            p1_stage_costs = [
                solver.stage_costs[0](state[:game.nx1], u[:game.nu1])
                for state, u in zip(states, raw_data.u)
            ]
        p1_costs_to_go = np.cumsum(p1_stage_costs[::-1])[::-1]
        raw_data.p1_total_cost = float(p1_costs_to_go[0])
        if stage_costs is None:
            p2_stage_costs = [
                float(solver.l2(state[game.nx1:], u[game.nu1:],
                                state[:game.nx1], u[:game.nu1]))
                for state, u in zip(states, raw_data.u)
            ]
        else:
            p2_stage_costs = [
                float(solver.stage_costs[1](
                    state[game.nx1:2 * game.nx1],
                    u[game.nu1:2 * game.nu1]))
                for state, u in zip(states, raw_data.u)
            ]
        p2_costs_to_go = np.cumsum(p2_stage_costs[::-1])[::-1]
        raw_data.p2_total_cost = float(p2_costs_to_go[0])
        if getattr(game, "n_players", 2) == 3:
            p3_stage_costs = [
                float(solver.l3(
                    state[2 * game.nx1:3 * game.nx1],
                    u[2 * game.nu1:3 * game.nu1],
                )) for state, u in zip(states, raw_data.u)
            ]
            p3_costs_to_go = np.cumsum(p3_stage_costs[::-1])[::-1]
            raw_data.p3_total_cost = float(p3_costs_to_go[0])
        else:
            p3_costs_to_go = [np.nan] * len(states)

        for t, state, u, p1_cost_to_go, p2_cost_to_go, p3_cost_to_go in zip(
            raw_data.t,
            states,
            raw_data.u,
            p1_costs_to_go,
            p2_costs_to_go,
            p3_costs_to_go,
        ):
            analyzed_data.t.append(t)
            analyzed_data.state.append(state)

            # analyzed_data.c.append(np.array([cx_min, cx_max, cy_min, cy_max]))
            analyzed_data.Cost2Go.append(p1_cost_to_go)
            analyzed_data.Cost2Go2.append(p2_cost_to_go)
            if getattr(game, "n_players", 2) == 3:
                analyzed_data.Cost2Go3.append(p3_cost_to_go)
            analyzed_data.u2.append(u[2:4])

    analyzed_data.n_data = len(analyzed_data.state)
    learned_data.AnalyzedData = analyzed_data


def record_learned_state(learned_data, game, iteration, alpha, feasible=True):
    if len(learned_data.RawData) < iteration + 1:
        learned_data.RawData.append(SimpleNamespace())
        alpha_value = np.asarray(alpha, dtype=float).reshape(-1)
        learned_data.RawData[iteration].alpha = (
            float(alpha_value[0]) if alpha_value.size == 1 else alpha_value.copy()
        )
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
    target3_state = (
        np.asarray(game.x3f, dtype=float).reshape(-1)
        if game.n_players == 3 else None
    )
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
    targets = [target1_state, target2_state]
    if target3_state is not None:
        if target3_state.shape != (game.nx1,):
            raise ValueError(f"x3f must have shape ({game.nx1},)")
        targets.append(target3_state)
    raw_data.x.append(np.concatenate(targets))
    raw_data.u.append(np.zeros(game.nu, dtype=float))
