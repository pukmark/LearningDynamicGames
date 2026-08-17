import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
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


def _predicted_player_distance(solution, dt):
    """Return time and inter-player distance predicted by one solver."""
    if solution is None or not hasattr(solution, "x1") or not hasattr(solution, "x2"):
        return np.empty(0), np.empty(0)

    predicted_x1 = np.asarray(solution.x1, dtype=float)
    predicted_x2 = np.asarray(solution.x2, dtype=float)
    if (
        predicted_x1.ndim != 2
        or predicted_x2.ndim != 2
        or predicted_x1.shape[0] == 0
        or predicted_x2.shape[0] == 0
        or predicted_x1.shape[1] < 2
        or predicted_x2.shape[1] < 2
    ):
        return np.empty(0), np.empty(0)

    # The shorter prediction is held at its terminal state. In this project
    # that extends P1 over P2's two additional prediction steps.
    prediction_length = max(predicted_x1.shape[0], predicted_x2.shape[0])
    x1_indices = np.minimum(np.arange(prediction_length), predicted_x1.shape[0] - 1)
    x2_indices = np.minimum(np.arange(prediction_length), predicted_x2.shape[0] - 1)
    distance = np.linalg.norm(
        predicted_x1[x1_indices, :2] - predicted_x2[x2_indices, :2], axis=1
    )
    prediction_time = (
        float(getattr(solution, "t", 0.0)) + np.arange(prediction_length) * dt
    )
    return prediction_time, distance


def _player1_executed_cost(states, inputs, game, solver):
    """Return P1's accumulated stage cost for executed steps this iteration."""
    if len(inputs) <= 1:
        return 0.0

    total_cost = 0.0
    for state, control in zip(states[:-2], inputs[:-1]):
        total_cost += float(solver.l1(state[:game.nx1], control[:game.nu1], state[game.nx1:], control[game.nu1:]))
    return total_cost


def _player2_completed_cost(iteration_data, game, solver):
    """Return P2's total cost for a completed iteration."""
    stored_cost = getattr(iteration_data, "p2_total_cost", np.nan)
    if np.isfinite(stored_cost):
        return float(stored_cost)

    states = getattr(iteration_data, "x", [])
    inputs = getattr(iteration_data, "u", [])
    if len(states) == 0 or len(inputs) == 0:
        return np.nan

    return float(
        sum(
            float(solver.l2(state[game.nx1:], control[game.nu1:]))
            for state, control in zip(states, inputs)
        )
    )


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
    plot_rows = 6 if game.is_single_integrator else 7
    fig = plt.figure(figsize=(13, 15 if game.is_single_integrator else 17))
    gs = fig.add_gridspec(plot_rows, 2, width_ratios=(2.0, 1.0))
    ax_xy = fig.add_subplot(gs[:-1, 0])
    ax_xpos = fig.add_subplot(gs[0, 1])
    ax_ypos = fig.add_subplot(gs[1, 1])
    ax_u = fig.add_subplot(gs[2, 1])
    ax_cost = fig.add_subplot(gs[-1, :])
    if game.is_single_integrator:
        ax_velocity = None
        ax_distance = fig.add_subplot(gs[3, 1])
        ax_bargaining = fig.add_subplot(gs[4, 1])
    else:
        ax_velocity = fig.add_subplot(gs[3, 1])
        ax_distance = fig.add_subplot(gs[4, 1])
        ax_bargaining = fig.add_subplot(gs[5, 1])
    ax_nash_product = ax_bargaining.twinx()

    lines = {}
    lines["p1_state"], = ax_xy.plot([], [], "C0-", label="P1 state")
    lines["p2_state"], = ax_xy.plot([], [], "C1-", label="P2 state")
    lines["p1_current"], = ax_xy.plot([], [], "C0o")
    lines["p2_current"], = ax_xy.plot([], [], "C1o")
    separation_circles = (
        Circle(
            (0.0, 0.0),
            radius=game.d_sep,
            fill=False,
            edgecolor="C0",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            label=r"$d_{\mathrm{sep}}$ diameter",
            visible=False,
        ),
        Circle(
            (0.0, 0.0),
            radius=game.d_sep,
            fill=False,
            edgecolor="C1",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            visible=False,
        ),
    )
    for separation_circle in separation_circles:
        ax_xy.add_patch(separation_circle)
    lines["p1_prediction"], = ax_xy.plot(
        [], [], "C0--", alpha=0.8, label="P1 prediction (Solver1)"
    )
    lines["p2_prediction"], = ax_xy.plot(
        [], [], "C1--", alpha=0.8, label="P2 prediction (Solver2)"
    )
    lines["p1_terminal_candidates"], = ax_xy.plot(
        [], [], "C0x", alpha=0.75, linestyle="none", label="P1 examined terminals"
    )
    lines["p2_terminal_candidates"], = ax_xy.plot(
        [], [], "C1x", alpha=0.75, linestyle="none", label="P2 examined terminals"
    )
    lines["p1_selected_terminal"], = ax_xy.plot(
        [], [], marker="*", color="C4", markersize=13, linestyle="none",
        label="P1 bargained terminal",
    )
    lines["p2_selected_terminal"], = ax_xy.plot(
        [], [], marker="*", color="C5", markersize=13, linestyle="none",
        label="P2 bargained terminal",
    )
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
    lines["x_joint_terminal_candidates"], = ax_xpos.plot(
        [], [], "C6x", markersize=7, linestyle="none", label="examined terminals"
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
    lines["y_joint_terminal_candidates"], = ax_ypos.plot(
        [], [], "C6x", markersize=7, linestyle="none", label="examined terminals"
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
        lines["p1_v_prediction"], = ax_velocity.plot(
            [], [], "C0--", alpha=0.8, label="P1 v prediction (Solver1)"
        )
        lines["p2_v_prediction"], = ax_velocity.plot(
            [], [], "C1--", alpha=0.8, label="P2 v prediction (Solver1)"
        )
        ax_velocity.axhline(
            np.sqrt(game.vx_max**2 + game.vy_max**2), color="C4", linestyle=":", linewidth=2,
            label="RSS maximum",
        )
        ax_velocity.set_xlabel("time")
        ax_velocity.set_ylabel("velocity")
        ax_velocity.set_title("Player velocities and root sum square")
        ax_velocity.grid(True, alpha=0.3)
        # ax_velocity.legend(loc="best", ncol=2)

    ax_cost.set_xlabel("iteration")
    ax_cost.set_ylabel("total cost-to-go")
    ax_cost.set_title("Player total cost-to-go by iteration")
    ax_cost.grid(True, axis="y", alpha=0.3)
    # ax_cost.legend(
    #     handles=(
    #         Patch(facecolor="C0", label="P1 completed"),
    #         Patch(facecolor="C1", label="P2 completed"),
    #         Patch(facecolor="C4", label="P1 current predicted"),
    #     ),
    #     loc="best",
    # )

    lines["player_distance"], = ax_distance.plot(
        [], [], "k-", linewidth=2, label="Executed distance"
    )
    lines["solver1_predicted_distance"], = ax_distance.plot(
        [], [], "C0--", linewidth=1.5, label="P1 solver prediction"
    )
    lines["solver2_predicted_distance"], = ax_distance.plot(
        [], [], "C1-.", linewidth=1.5, label="P2 solver prediction"
    )
    ax_distance.axhline(
        game.d_sep, color="C3", linestyle=":", linewidth=1.5,
        label="Minimum separation",
    )
    ax_distance.set_xlabel("time")
    ax_distance.set_ylabel("distance")
    ax_distance.set_title("Distance between players")
    ax_distance.grid(True, alpha=0.3)
    # ax_distance.legend(loc="best")

    lines["bargaining_gamma"], = ax_bargaining.plot(
        [], [], "C4-o", markersize=3, linewidth=1.5, label=r"chosen $\gamma^*$"
    )
    lines["nash_product"], = ax_nash_product.plot(
        [], [], "C2--", linewidth=1.5, label=r"$\Delta_1\Delta_2$"
    )
    bargaining_text = ax_bargaining.text(
        0.02, 0.04, "No bargaining agreement yet",
        transform=ax_bargaining.transAxes,
        va="bottom", ha="left", fontsize=8, family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82},
    )
    ax_bargaining.set_xlabel("time")
    ax_bargaining.set_ylabel(r"$\gamma^*$", color="C4")
    ax_bargaining.set_ylim(-0.05, 1.05)
    ax_bargaining.tick_params(axis="y", labelcolor="C4")
    ax_bargaining.set_title("Cooperative Nash bargaining")
    ax_bargaining.grid(True, alpha=0.3)
    ax_nash_product.set_ylabel("Nash product", color="C2")
    ax_nash_product.tick_params(axis="y", labelcolor="C2")

    fig.tight_layout()
    state = {
        "fig": fig,
        "ax_xy": ax_xy,
        "ax_xpos": ax_xpos,
        "ax_ypos": ax_ypos,
        "ax_u": ax_u,
        "ax_velocity": ax_velocity,
        "ax_cost": ax_cost,
        "ax_distance": ax_distance,
        "ax_bargaining": ax_bargaining,
        "ax_nash_product": ax_nash_product,
        "bargaining_text": bargaining_text,
        "lines": lines,
        "separation_circles": separation_circles,
        "iteration": game.iteration,
        "past_xy_lines": [],
        "cost_bars": [],
        "cost_labels": [],
        "plotted_iteration_costs": None,
        "predicted_cost1": None,
        "bargaining_times": [],
        "bargaining_gammas": [],
        "nash_products": [],
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
    ax_distance = state["ax_distance"]
    ax_bargaining = state["ax_bargaining"]
    ax_nash_product = state["ax_nash_product"]

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
        state["bargaining_times"].clear()
        state["bargaining_gammas"].clear()
        state["nash_products"].clear()

    lines["p1_state"].set_data(x[:, 0], x[:, 1])
    lines["p2_state"].set_data(x[:, p2_i], x[:, p2_i + 1])
    lines["p1_current"].set_data([x[-1, 0]], [x[-1, 1]])
    lines["p2_current"].set_data([x[-1, p2_i]], [x[-1, p2_i + 1]])
    player_positions = (
        (x[-1, 0], x[-1, 1]),
        (x[-1, p2_i], x[-1, p2_i + 1]),
    )
    for separation_circle, position in zip(
        state["separation_circles"], player_positions
    ):
        separation_circle.center = position
        separation_circle.set_radius(game.d_sep / 2)
        separation_circle.set_visible(True)
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
    solver2_solution = getattr(solver2, "Solution", None)

    try:
        bargaining_gamma = float(getattr(solution, "bargaining_gamma", np.nan))
    except (TypeError, ValueError):
        bargaining_gamma = np.nan
    try:
        nash_product = float(getattr(solution, "nash_product", np.nan))
    except (TypeError, ValueError):
        nash_product = np.nan
    if np.isfinite(bargaining_gamma) and np.isfinite(nash_product):
        agreement_time = float(getattr(solution, "t", game.t))
        if (
            state["bargaining_times"]
            and np.isclose(state["bargaining_times"][-1], agreement_time)
        ):
            state["bargaining_gammas"][-1] = float(bargaining_gamma)
            state["nash_products"][-1] = float(nash_product)
        else:
            state["bargaining_times"].append(agreement_time)
            state["bargaining_gammas"].append(float(bargaining_gamma))
            state["nash_products"].append(float(nash_product))

    lines["bargaining_gamma"].set_data(
        state["bargaining_times"], state["bargaining_gammas"]
    )
    lines["nash_product"].set_data(
        state["bargaining_times"], state["nash_products"]
    )
    if state["bargaining_times"]:
        ax_bargaining.set_xlim(
            min(0.0, state["bargaining_times"][0]),
            max(game.dt, game.t, state["bargaining_times"][-1] + game.dt),
        )
        ax_nash_product.relim()
        ax_nash_product.autoscale_view(scalex=False)

    baseline = np.asarray(
        getattr(solution, "disagreement_costs", []), dtype=float
    ).reshape(-1)
    improvements = np.asarray(
        getattr(solution, "bargaining_improvements", []), dtype=float
    ).reshape(-1)
    costs = np.asarray(
        [
            getattr(solution, "player1_cost", np.nan),
            getattr(solution, "player2_cost", np.nan),
        ],
        dtype=float,
    )
    terminal_index = getattr(solution, "terminal_sample_index", None)
    terminal_time = getattr(solution, "terminal_sample_time", np.nan)
    if (
        np.isfinite(bargaining_gamma)
        and np.isfinite(nash_product)
        and baseline.shape == (2,)
        and improvements.shape == (2,)
        and np.all(np.isfinite(costs))
    ):
        state["bargaining_text"].set_text(
            f"z*: sample {terminal_index}, safe t={terminal_time:.2f}\n"
            f"gamma*={float(bargaining_gamma):.3f}   Nash={float(nash_product):.3g}\n"
            f"C=({costs[0]:.3g}, {costs[1]:.3g})\n"
            f"b=({baseline[0]:.3g}, {baseline[1]:.3g})\n"
            f"Delta=({improvements[0]:.3g}, {improvements[1]:.3g})"
        )
    else:
        state["bargaining_text"].set_text("No bargaining agreement yet")

    candidate_terminal_states = np.asarray(
        getattr(solution, "candidate_terminal_states", []), dtype=float
    )
    if (
        candidate_terminal_states.ndim == 2
        and candidate_terminal_states.shape[1] == game.nx
    ):
        candidate_x_positions = candidate_terminal_states[:, [0, p2_i]]
        candidate_y_positions = candidate_terminal_states[:, [1, p2_i + 1]]
        lines["p1_terminal_candidates"].set_data(
            candidate_terminal_states[:, 0], candidate_terminal_states[:, 1]
        )
        lines["p2_terminal_candidates"].set_data(
            candidate_terminal_states[:, p2_i],
            candidate_terminal_states[:, p2_i + 1],
        )
        lines["x_joint_terminal_candidates"].set_data(
            candidate_x_positions[:, 0], candidate_x_positions[:, 1]
        )
        lines["y_joint_terminal_candidates"].set_data(
            candidate_y_positions[:, 0], candidate_y_positions[:, 1]
        )
    else:
        candidate_terminal_states = np.empty((0, game.nx))
        candidate_x_positions = np.empty((0, 2))
        candidate_y_positions = np.empty((0, 2))
        lines["p1_terminal_candidates"].set_data([], [])
        lines["p2_terminal_candidates"].set_data([], [])
        lines["x_joint_terminal_candidates"].set_data([], [])
        lines["y_joint_terminal_candidates"].set_data([], [])

    selected_terminal = np.asarray(
        getattr(solution, "terminal_sample_state", []), dtype=float
    ).reshape(-1)
    if selected_terminal.shape == (game.nx,) and np.isfinite(bargaining_gamma):
        lines["p1_selected_terminal"].set_data(
            [selected_terminal[0]], [selected_terminal[1]]
        )
        lines["p2_selected_terminal"].set_data(
            [selected_terminal[p2_i]], [selected_terminal[p2_i + 1]]
        )
    else:
        lines["p1_selected_terminal"].set_data([], [])
        lines["p2_selected_terminal"].set_data([], [])

    raw_data = getattr(learned_data, "RawData", [])
    completed_iteration_costs = tuple(
        (
            iteration_index + 1,
            float(iteration_data.p1_total_cost),
            _player2_completed_cost(iteration_data, game, solver2),
        )
        for iteration_index, iteration_data in enumerate(raw_data)
        if np.isfinite(getattr(iteration_data, "p1_total_cost", np.nan))
    )
    plotted_costs = completed_iteration_costs
    predicted_cost_to_go = getattr(solution, "player1_cost", np.nan)
    if (
        bool(getattr(solution, "success", False))
        and np.isfinite(predicted_cost_to_go)
        and game.iteration not in {item[0] for item in completed_iteration_costs}
    ):
        predicted_iteration_cost = _player1_executed_cost(x, u, game, solver1)
        predicted_iteration_cost += float(predicted_cost_to_go)
        predicted_cost = (game.iteration, predicted_iteration_cost)
        state["predicted_cost1"] = predicted_cost
    else:
        predicted_cost = state["predicted_cost1"]

    cost_plot_data = (plotted_costs, predicted_cost)
    if cost_plot_data != state["plotted_iteration_costs"]:
        for label in state["cost_labels"]:
            label.remove()
        state["cost_labels"] = []
        for bars in state["cost_bars"]:
            bars.remove()
        state["cost_bars"] = []

        completed_iterations = [item[0] for item in completed_iteration_costs]
        p1_completed_values = [item[1] for item in completed_iteration_costs]
        p2_completed_values = [item[2] for item in completed_iteration_costs]
        bar_width = 0.34
        if completed_iterations:
            p1_bars = ax_cost.bar(
                np.asarray(completed_iterations) - bar_width / 2,
                p1_completed_values,
                color="C0",
                width=bar_width,
            )
            p2_bars = ax_cost.bar(
                np.asarray(completed_iterations) + bar_width / 2,
                p2_completed_values,
                color="C1",
                width=bar_width,
            )
            state["cost_bars"].extend((p1_bars, p2_bars))

        if predicted_cost is not None:
            predicted_bars = ax_cost.bar(
                [predicted_cost[0]],
                [predicted_cost[1]],
                color="C4",
                width=0.7,
            )
            state["cost_bars"].append(predicted_bars)

        for bars in state["cost_bars"]:
            cost_labels = [
                f"{bar.get_height():.2f}".rstrip("0").rstrip(".")
                for bar in bars
            ]
            state["cost_labels"].extend(
                ax_cost.bar_label(bars, labels=cost_labels, padding=3)
            )

        state["plotted_iteration_costs"] = cost_plot_data
        plotted_iterations = completed_iterations.copy()
        if predicted_cost is not None:
            plotted_iterations.append(predicted_cost[0])
        plotted_iterations = sorted(set(plotted_iterations))
        if plotted_iterations:
            ax_cost.set_xticks(plotted_iterations)
            ax_cost.set_xlim(0.4, plotted_iterations[-1] + 0.6)
        ax_cost.relim()
        ax_cost.autoscale_view(scalex=False)
        ax_cost.margins(y=0.12)

    executed_distance = np.linalg.norm(
        x[:, :2] - x[:, p2_i:p2_i + 2], axis=1
    )
    lines["player_distance"].set_data(t, executed_distance)

    solver1_prediction_time, solver1_predicted_distance = (
        _predicted_player_distance(solution, game.dt)
    )
    lines["solver1_predicted_distance"].set_data(
        solver1_prediction_time, solver1_predicted_distance
    )
    solver2_prediction_time, solver2_predicted_distance = (
        _predicted_player_distance(solver2_solution, game.dt)
    )
    lines["solver2_predicted_distance"].set_data(
        solver2_prediction_time, solver2_predicted_distance
    )
    ax_distance.relim()
    ax_distance.autoscale_view()
    ax_distance.set_ylim(bottom=game.d_sep - 0.1, top=game.d_sep + 0.25)

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
        (
            x[:, [0, p2_i]],
            sampled_x_positions,
            candidate_x_positions,
            joint_x_target,
        )
    )
    joint_y_positions = np.vstack(
        (
            x[:, [1, p2_i + 1]],
            sampled_y_positions,
            candidate_y_positions,
            joint_y_target,
        )
    )
    _set_tight_joint_limits(ax_xpos, joint_x_positions)
    _set_tight_joint_limits(ax_ypos, joint_y_positions)

    if solution is not None and hasattr(solution, "x1"):
        lines["p1_prediction"].set_data(solution.x1[:, 0], solution.x1[:, 1])
    else:
        lines["p1_prediction"].set_data([], [])

    if solver2_solution is not None and hasattr(solver2_solution, "x2"):
        lines["p2_prediction"].set_data(
            solver2_solution.x2[:, 0], solver2_solution.x2[:, 1]
        )
    else:
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

        lines["p1_v"].set_data(t, np.sqrt(np.sum(p1_velocity**2, axis=1)))
        lines["p2_v"].set_data(t, np.sqrt(np.sum(p2_velocity**2, axis=1)))

        if solution is not None and hasattr(solution, "x1") and hasattr(solution, "x2"):
            predicted_x1 = np.asarray(solution.x1, dtype=float)
            predicted_x2 = np.asarray(solution.x2, dtype=float)
            if (
                predicted_x1.ndim == 2
                and predicted_x2.ndim == 2
                and predicted_x1.shape[1] >= 4
                and predicted_x2.shape[1] >= 4
            ):
                p1_prediction_time = (
                    float(getattr(solution, "t", game.t))
                    + np.arange(predicted_x1.shape[0]) * game.dt
                )
                p2_prediction_time = (
                    float(getattr(solution, "t", game.t))
                    + np.arange(predicted_x2.shape[0]) * game.dt
                )
                lines["p1_v_prediction"].set_data(
                    p1_prediction_time,
                    np.linalg.norm(predicted_x1[:, 2:4], axis=1),
                )
                lines["p2_v_prediction"].set_data(
                    p2_prediction_time,
                    np.linalg.norm(predicted_x2[:, 2:4], axis=1),
                )
            else:
                lines["p1_v_prediction"].set_data([], [])
                lines["p2_v_prediction"].set_data([], [])
        else:
            lines["p1_v_prediction"].set_data([], [])
            lines["p2_v_prediction"].set_data([], [])

        ax_velocity.relim()
        ax_velocity.autoscale_view()
        
    equilibrium_label = (
        rf"$\gamma^*$={float(bargaining_gamma):.2f}"
        if np.isfinite(bargaining_gamma)
        else f"alpha1={solver1.alpha_vec[0,0]:.2f}, alpha2={solver2.alpha_vec[0,0]:.2f}"
    )
    ax_xy.set_title(
        f"XY trajectory - Iteration: {game.iteration}, {equilibrium_label}, "
        f"time: {game.t:.2f}"
    )

    fig.canvas.draw_idle()
    if plt.get_backend().lower() != "agg" and pause is not None:
        plt.pause(pause)
    return fig
