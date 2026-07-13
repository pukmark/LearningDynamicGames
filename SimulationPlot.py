import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull, QhullError

eps = 1e-1


def _set_joint_sample_plot(sample_line, hull_line, points):
    if points.size == 0:
        sample_line.set_data([], [])
        hull_line.set_data([], [])
        return

    sample_line.set_data(points[:, 0], points[:, 1])

    unique_points = np.unique(points, axis=0)
    if unique_points.shape[0] < 3 or np.linalg.matrix_rank(unique_points - unique_points[0]) < 2:
        hull_line.set_data([], [])
        return

    try:
        hull = ConvexHull(unique_points)
    except QhullError:
        hull_line.set_data([], [])
        return

    hull_points = unique_points[np.r_[hull.vertices, hull.vertices[0]]]
    hull_line.set_data(hull_points[:, 0], hull_points[:, 1])


def _set_tight_joint_limits(ax, points, margin_fraction=0.08):
    finite_points = points[np.isfinite(points).all(axis=1)]
    if finite_points.size == 0:
        return

    for set_limits, values in (
        (ax.set_xlim, finite_points[:, 0]),
        (ax.set_ylim, finite_points[:, 1]),
    ):
        value_min = values.min()
        value_max = values.max()
        span = value_max - value_min
        padding = span * margin_fraction if span > 0 else max(eps, abs(value_min) * margin_fraction)
        set_limits(value_min - padding, value_max + padding)


def _player1_executed_cost(states, inputs, game, solver):
    """Return P1's accumulated stage cost for executed steps this iteration."""
    if len(inputs) <= 1:
        return 0.0

    total_cost = 0.0
    for state, control in zip(states[:-2], inputs[:-1]):
        total_cost += float(solver.l1(state[:game.nx1], control[:game.nu1]))
    return total_cost


def close_simulation_plots():
    """Clear plot state and close all matplotlib figures."""
    state = getattr(plot_simulation, "_state", None)
    if state is not None:
        state["fig"].clf()
        plot_simulation._state = None
    plt.close("all")


def save_simulation_figure(path="LDG_Simulation.png"):
    """Save the current simulation figure to path."""
    state = getattr(plot_simulation, "_state", None)
    if state is None:
        raise RuntimeError("Simulation plot has not been initialized")

    figure = state["fig"]
    figure.canvas.draw()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    return path


def plot_simulation_init(game):
    plt.ion()
    plot_rows = 5 if game.is_single_integrator else 6
    fig = plt.figure(figsize=(13, 13 if game.is_single_integrator else 15))
    gs = fig.add_gridspec(plot_rows, 2, width_ratios=(2.0, 1.0))
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_xpos = fig.add_subplot(gs[0, 1])
    ax_ypos = fig.add_subplot(gs[1, 1])
    ax_u = fig.add_subplot(gs[2, 1])
    if game.is_single_integrator:
        ax_velocity = None
        ax_cost = fig.add_subplot(gs[3, 1])
        ax_arrival = fig.add_subplot(gs[4, 1])
    else:
        ax_velocity = fig.add_subplot(gs[3, 1])
        ax_cost = fig.add_subplot(gs[4, 1])
        ax_arrival = fig.add_subplot(gs[5, 1])

    lines = {}
    lines["p1_state"], = ax_xy.plot([], [], "C0-", label="P1 state")
    lines["p2_state"], = ax_xy.plot([], [], "C1-", label="P2 state")
    lines["p1_current"], = ax_xy.plot([], [], "C0o")
    lines["p2_current"], = ax_xy.plot([], [], "C1o")
    lines["p1_prediction"], = ax_xy.plot([], [], "C0--", alpha=0.8, label="P1 prediction")
    lines["p2_prediction"], = ax_xy.plot([], [], "C1--", alpha=0.8, label="P2 prediction")
    lines["Target1"], = ax_xy.plot([], [], "ks", alpha=1.0, label="Target 1", linewidth=3)
    lines["Target2"], = ax_xy.plot([], [], "ks", alpha=1.0, label="Target 2", linewidth=3)
    ax_xy.axhline(game.y_min, color="0.75", linewidth=0.8)
    ax_xy.axhline(game.y_max, color="0.75", linewidth=0.8)
    ax_xy.set_xlim(game.x_min-eps, game.x_max+eps)
    ax_xy.set_ylim(game.y_min-eps, game.y_max+eps)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x position")
    ax_xy.set_ylabel("y position")
    ax_xy.set_title(f"XY trajectory - Iteration: {game.iteration}")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

    lines["x_joint_state"], = ax_xpos.plot([], [], color="C2", linestyle="-", marker='s', label="current trajectory")
    lines["x_joint_current"], = ax_xpos.plot([], [], "C2o", label="current")
    lines["x_joint_target"], = ax_xpos.plot([], [], "ks", markersize=7, label="target")
    lines["x_joint_samples"], = ax_xpos.plot([], [], "k.", alpha=0.35, label="sampled set")
    lines["x_joint_hull"], = ax_xpos.plot([], [], "k-", linewidth=1.5, label="sampled convex hull")
    lines["x_joint_active_samples"], = ax_xpos.plot(
        [], [], "o", color="C4", markerfacecolor="none", markersize=8,
        markeredgewidth=2, linestyle="none", label="solution data",
    )
    ax_xpos.set_xlim(game.x_min-eps, game.x_max+eps)
    ax_xpos.set_ylim(game.x_min-eps, game.x_max+eps)
    ax_xpos.set_aspect("equal", adjustable="box")
    ax_xpos.set_xlabel("P1 x position")
    ax_xpos.set_ylabel("P2 x position")
    ax_xpos.set_title("Joint x position")
    ax_xpos.grid(True, alpha=0.3)
    # ax_xpos.legend(loc="best")

    lines["y_joint_state"], = ax_ypos.plot([], [], color="C3", linestyle="-", marker='s', label="current trajectory")
    lines["y_joint_current"], = ax_ypos.plot([], [], "C3o", label="current")
    lines["y_joint_target"], = ax_ypos.plot([], [], "ks", markersize=7, label="target")
    lines["y_joint_samples"], = ax_ypos.plot([], [], "k.", alpha=0.35, label="sampled set")
    lines["y_joint_hull"], = ax_ypos.plot([], [], "k-", linewidth=1.5, label="sampled convex hull")
    lines["y_joint_active_samples"], = ax_ypos.plot(
        [], [], "o", color="C4", markerfacecolor="none", markersize=8,
        markeredgewidth=2, linestyle="none", label="solution data",
    )
    ax_ypos.set_xlim(game.y_min-eps, game.y_max+eps)
    ax_ypos.set_ylim(game.y_min-eps, game.y_max+eps)
    ax_ypos.set_aspect("equal", adjustable="box")
    ax_ypos.set_xlabel("P1 y position")
    ax_ypos.set_ylabel("P2 y position")
    ax_ypos.set_title("Joint y position")
    ax_ypos.grid(True, alpha=0.3)
    # ax_ypos.legend(loc="best")

    input_label = "v" if game.is_single_integrator else "a"
    lines["p1_ax"], = ax_u.plot([], [], color="C0", linestyle="-", drawstyle="steps-post", label=f"P1 {input_label}x")
    lines["p1_ay"], = ax_u.plot([], [], color="C0", linestyle="--", drawstyle="steps-post", label=f"P1 {input_label}y")
    lines["p2_ax"], = ax_u.plot([], [], color="C1", linestyle="-", drawstyle="steps-post", label=f"P2 {input_label}x")
    lines["p2_ay"], = ax_u.plot([], [], color="C1", linestyle="--", drawstyle="steps-post", label=f"P2 {input_label}y")
    lines["sum_ax"], = ax_u.plot([], [], color="C2", linestyle="-", linewidth=2, drawstyle="steps-post", label=f"Sum {input_label}x")
    lines["sum_ay"], = ax_u.plot([], [], color="C3", linestyle="--", linewidth=2, drawstyle="steps-post", label=f"Sum {input_label}y")
    ax_u.axhline(game.u_max_shared, color="C4", linestyle=":", linewidth=2, label="Shared input maximum")
    ax_u.axhline(game.u_min_shared, color="C4", linestyle=":", linewidth=2, label="Shared input minimum")
    ax_u.set_xlabel("time")
    ax_u.set_ylabel("input")
    ax_u.set_title("Inputs vs time")
    ax_u.grid(True, alpha=0.3)
    # ax_u.legend(loc="best", ncol=2)

    if ax_velocity is not None:
        lines["p1_v"], = ax_velocity.plot([], [], "C0-", label="P1 v")
        lines["p2_v"], = ax_velocity.plot([], [], "C1-", label="P2 v")
        lines["velocity_rss"], = ax_velocity.plot(
            [], [], "C2-", linewidth=2, label="velocity RSS"
        )
        ax_velocity.axhline(
            game.vx_max, color="C4", linestyle=":", linewidth=2,
            label="RSS maximum",
        )
        ax_velocity.set_xlabel("time")
        ax_velocity.set_ylabel("velocity")
        ax_velocity.set_title("Player velocities and root sum square")
        ax_velocity.grid(True, alpha=0.3)
        ax_velocity.legend(loc="best", ncol=2)

    ax_cost.set_xlabel("iteration")
    ax_cost.set_ylabel("total cost-to-go")
    ax_cost.set_title("P1 total cost-to-go by iteration")
    ax_cost.grid(True, axis="y", alpha=0.3)
    # ax_cost.legend(
    #     handles=(
    #         Patch(facecolor="C5", label="completed total cost"),
    #         Patch(facecolor="C4", label="current predicted iteration total"),
    #     ),
    #     loc="best",
    # )

    lines["p1_arrival_time"], = ax_arrival.plot(
        [], [], "C0o-", label="P1 arrival time"
    )
    lines["p2_arrival_time"], = ax_arrival.plot(
        [], [], "C1o-", label="P2 arrival time"
    )
    lines["target_arrival_time"], = ax_arrival.plot(
        [], [], "C2o-", label="Target arrival time"
    )
    ax_arrival.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax_arrival.set_xlabel("completed iteration")
    ax_arrival.set_ylabel("arrival time")
    ax_arrival.set_title("Target arrival times")
    ax_arrival.grid(True, alpha=0.3)
    ax_arrival.legend(loc="best")

    fig.tight_layout()
    state = {
        "fig": fig,
        "ax_xy": ax_xy,
        "ax_xpos": ax_xpos,
        "ax_ypos": ax_ypos,
        "ax_u": ax_u,
        "ax_velocity": ax_velocity,
        "ax_cost": ax_cost,
        "ax_arrival": ax_arrival,
        "lines": lines,
        "iteration": game.iteration,
        "past_xy_lines": [],
        "cost_bars": None,
        "cost_labels": [],
        "plotted_iteration_costs": None,
    }
    plot_simulation._state = state
    if plt.get_backend().lower() != "agg":
        plt.pause(1.0)

def plot_simulation(game, solver1, solver2, LearnedData, pause=0.01):
    """Update a realtime plot for the current game and solver state."""

    state = getattr(plot_simulation, "_state", None)

    fig = state["fig"]
    ax_u = state["ax_u"]
    ax_xy = state["ax_xy"]
    lines = state["lines"]
    ax_xpos = state["ax_xpos"]
    ax_ypos = state["ax_ypos"]
    ax_velocity = state["ax_velocity"]
    ax_cost = state["ax_cost"]
    ax_arrival = state["ax_arrival"]

    history = game.get_history()
    t = history["t"]
    x = history["x"]
    u = history["u"]
    p2_i = game.nx1

    if game.iteration != state["iteration"]:
        p1_x, p1_y = lines["p1_state"].get_data()
        p2_x, p2_y = lines["p2_state"].get_data()
        if len(p1_x) > 1:
            past_p1, = ax_xy.plot(
                np.asarray(p1_x).copy(),
                np.asarray(p1_y).copy(),
                color="C0",
                marker='o',
                linewidth=1.0,
                alpha=0.1,
                zorder=1,
            )
            past_p2, = ax_xy.plot(
                np.asarray(p2_x).copy(),
                np.asarray(p2_y).copy(),
                color="C1",
                marker='o',
                linewidth=1.0,
                alpha=0.1,
                zorder=1,
                )
            state["past_xy_lines"].extend((past_p1, past_p2))
        state["iteration"] = game.iteration

    lines["p1_state"].set_data(x[:, 0], x[:, 1])
    lines["p2_state"].set_data(x[:, p2_i], x[:, p2_i + 1])
    lines["p1_current"].set_data([x[-1, 0]], [x[-1, 1]])
    lines["p2_current"].set_data([x[-1, p2_i]], [x[-1, p2_i + 1]])
    lines["x_joint_state"].set_data(x[:, 0], x[:, p2_i])
    lines["x_joint_current"].set_data([x[-1, 0]], [x[-1, p2_i]])
    lines["y_joint_state"].set_data(x[:, 1], x[:, p2_i + 1])
    lines["y_joint_current"].set_data([x[-1, 1]], [x[-1, p2_i + 1]])

    target1_position = np.asarray(game.x1f, dtype=float).reshape(-1)[:2]
    target2_position = np.asarray(game.x2f, dtype=float).reshape(-1)[:2]
    joint_x_target = np.array([[target1_position[0], target2_position[0]]])
    joint_y_target = np.array([[target1_position[1], target2_position[1]]])
    lines["x_joint_target"].set_data(joint_x_target[:, 0], joint_x_target[:, 1])
    lines["y_joint_target"].set_data(joint_y_target[:, 0], joint_y_target[:, 1])

    learned_data = LearnedData
    analyzed_data = learned_data.AnalyzedData
    sampled_states = analyzed_data.state
    solution = getattr(solver1, "Solution", None)

    raw_data = getattr(learned_data, "RawData", [])
    completed_iteration_costs = tuple(
        (iteration_index + 1, float(iteration_data.p1_total_cost))
        for iteration_index, iteration_data in enumerate(raw_data)
        if np.isfinite(getattr(iteration_data, "p1_total_cost", np.nan))
    )
    plotted_costs = tuple(
        (iteration, cost, False)
        for iteration, cost in completed_iteration_costs
    )
    predicted_cost_to_go = getattr(solution, "player1_cost", np.nan)
    if (
        bool(getattr(solution, "success", False))
        and np.isfinite(predicted_cost_to_go)
        and game.iteration not in {item[0] for item in completed_iteration_costs}
    ):
        predicted_iteration_cost = _player1_executed_cost(x, u, game, solver1)
        predicted_iteration_cost += float(predicted_cost_to_go)
        plotted_costs += ((game.iteration, predicted_iteration_cost, True),)

    if plotted_costs != state["plotted_iteration_costs"]:
        for label in state["cost_labels"]:
            label.remove()
        state["cost_labels"] = []
        if state["cost_bars"] is not None:
            state["cost_bars"].remove()

        plotted_iterations = [item[0] for item in plotted_costs]
        plotted_values = [item[1] for item in plotted_costs]
        bar_colors = ["C4" if item[2] else "C5" for item in plotted_costs]
        state["cost_bars"] = ax_cost.bar(
            plotted_iterations,
            plotted_values,
            color=bar_colors,
            width=0.7,
        )
        cost_labels = [f"{value:.2f}".rstrip("0").rstrip(".") for value in plotted_values]
        state["cost_labels"] = ax_cost.bar_label(
            state["cost_bars"],
            labels=cost_labels,
            padding=3,
        )
        state["plotted_iteration_costs"] = plotted_costs
        if plotted_iterations:
            ax_cost.set_xticks(plotted_iterations)
            ax_cost.set_xlim(0.4, plotted_iterations[-1] + 0.6)
        ax_cost.relim()
        ax_cost.autoscale_view(scalex=False)
        ax_cost.margins(y=0.12)

    p1_completed_iterations = []
    p1_arrival_times = []
    p2_completed_iterations = []
    p2_arrival_times = []
    for iteration_index, iteration_data in enumerate(raw_data):
        p1_arrival_time = getattr(iteration_data, "p1_arrival_time", np.nan)
        p2_arrival_time = getattr(iteration_data, "p2_arrival_time", np.nan)
        if np.isfinite(p1_arrival_time):
            p1_completed_iterations.append(iteration_index + 1)
            p1_arrival_times.append(p1_arrival_time)
        if np.isfinite(p2_arrival_time):
            p2_completed_iterations.append(iteration_index + 1)
            p2_arrival_times.append(p2_arrival_time)
    lines["p1_arrival_time"].set_data(
        p1_completed_iterations, p1_arrival_times
    )
    lines["p2_arrival_time"].set_data(
        p2_completed_iterations, p2_arrival_times
    )
    ax_arrival.relim()
    ax_arrival.autoscale_view()

    if len(sampled_states) > 0:
        sampled_states = np.asarray(sampled_states, dtype=float)
        sampled_x_positions = sampled_states[:, [0, p2_i]]
        sampled_y_positions = sampled_states[:, [1, p2_i + 1]]
        _set_joint_sample_plot(
            lines["x_joint_samples"],
            lines["x_joint_hull"],
            sampled_x_positions,
        )
        _set_joint_sample_plot(
            lines["y_joint_samples"],
            lines["y_joint_hull"],
            sampled_y_positions,
        )

        a_set = np.asarray(getattr(solution, "a_set", []), dtype=float).reshape(-1)
        if a_set.size == sampled_states.shape[0]:
            active_samples = a_set > 1e-3
            lines["x_joint_active_samples"].set_data(
                sampled_x_positions[active_samples, 0],
                sampled_x_positions[active_samples, 1],
            )
            lines["y_joint_active_samples"].set_data(
                sampled_y_positions[active_samples, 0],
                sampled_y_positions[active_samples, 1],
            )
        else:
            lines["x_joint_active_samples"].set_data([], [])
            lines["y_joint_active_samples"].set_data([], [])
    else:
        sampled_x_positions = np.empty((0, 2))
        sampled_y_positions = np.empty((0, 2))
        _set_joint_sample_plot(lines["x_joint_samples"], lines["x_joint_hull"], np.empty((0, 2)))
        _set_joint_sample_plot(lines["y_joint_samples"], lines["y_joint_hull"], np.empty((0, 2)))
        lines["x_joint_active_samples"].set_data([], [])
        lines["y_joint_active_samples"].set_data([], [])

    joint_x_positions = np.vstack(
        (x[:, [0, p2_i]], sampled_x_positions, joint_x_target)
    )
    joint_y_positions = np.vstack(
        (x[:, [1, p2_i + 1]], sampled_y_positions, joint_y_target)
    )
    _set_tight_joint_limits(ax_xpos, joint_x_positions)
    _set_tight_joint_limits(ax_ypos, joint_y_positions)

    if (
        solution is not None
        and hasattr(solution, "x1")
        and hasattr(solution, "x2")
    ):
        lines["p1_prediction"].set_data(solution.x1[:, 0], solution.x1[:, 1])
        lines["p2_prediction"].set_data(solution.x2[:, 0], solution.x2[:, 1])
    else:
        lines["p1_prediction"].set_data([], [])
        lines["p2_prediction"].set_data([], [])
    lines["Target1"].set_data([game.x1f[0,0]],[game.x1f[0,1]])
    lines["Target2"].set_data([game.x2f[0,0]],[game.x2f[0,1]])

    valid_u = np.isfinite(u).all(axis=1)
    if np.any(valid_u):
        tu = t[:-1][valid_u]
        uu = u[valid_u]
        lines["p1_ax"].set_data(tu, uu[:, 0])
        lines["p1_ay"].set_data(tu, uu[:, 1])
        lines["p2_ax"].set_data(tu, uu[:, 2])
        lines["p2_ay"].set_data(tu, uu[:, 3])
        lines["sum_ax"].set_data(tu, uu[:, 0] + uu[:, 2])
        lines["sum_ay"].set_data(tu, uu[:, 1] + uu[:, 3])
    else:
        lines["p1_ax"].set_data([], [])
        lines["p1_ay"].set_data([], [])
        lines["p2_ax"].set_data([], [])
        lines["p2_ay"].set_data([], [])
        lines["sum_ax"].set_data([], [])
        lines["sum_ay"].set_data([], [])

    ax_u.relim()
    ax_u.autoscale_view()

    if ax_velocity is not None:
        p1_velocity = x[:, 2:4]
        p2_velocity = x[:, p2_i + 2:p2_i + 4]
        velocity_rss = np.sqrt(
            np.sum(p1_velocity**2, axis=1) + np.sum(p2_velocity**2, axis=1)
        )
        lines["p1_v"].set_data(t, np.sqrt(np.sum(p1_velocity**2, axis=1)))
        lines["p2_v"].set_data(t, np.sqrt(np.sum(p2_velocity**2, axis=1)))
        lines["velocity_rss"].set_data(t, velocity_rss)
        ax_velocity.relim()
        ax_velocity.autoscale_view()
        
    ax_xy.set_title(f"XY trajectory - Iteration: {game.iteration}, alpha1={solver1.alpha_vec[0,0]:2.2}, alpha2={solver2.alpha_vec[0,0]:2.2}, time: {game.t:2.2}")

    fig.canvas.draw_idle()
    if plt.get_backend().lower() != "agg" and pause is not None:
        plt.pause(pause)
    return fig
