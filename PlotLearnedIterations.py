"""Plot selected learning iterations from a saved LearnedData.pkl file.

Example:
    .venv/bin/python PlotLearnedIterations.py --iterations 1 3 5 10

Iteration numbers start at 1. Each panel includes all earlier paths with low
opacity. Saved total costs appear in each panel's legend.
The 7.16-inch-wide figure spans both columns of a paper, with trajectories in
one row. PNG output also saves a companion vector PDF.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator

from LDG_Simulation_aux import load_learned_data


def _plot_data(learned_data):
    raw = learned_data.RawData
    if not raw:
        raise ValueError("LearnedData contains no iterations")
    controls = np.asarray(raw[0].u, dtype=float)
    if controls.ndim != 2 or controls.shape[1] not in (4, 6):
        raise ValueError("expected four or six input columns for two or three players")
    players = controls.shape[1] // 2
    paths = [np.asarray(iteration.x, dtype=float) for iteration in raw]
    if paths[0].ndim != 2 or paths[0].shape[1] % players:
        raise ValueError("state columns do not match the number of players")
    nx = paths[0].shape[1] // players
    if nx not in (2, 3, 4):
        raise ValueError("expected two, three, or four states per player")
    for number, path in enumerate(paths, start=1):
        if path.ndim != 2 or path.shape[1] != players * nx or len(path) == 0:
            raise ValueError(f"iteration {number} has an empty or inconsistent state history")
        if not np.all(np.isfinite(path)):
            raise ValueError(f"iteration {number} contains nonfinite states")
        inputs = np.asarray(raw[number - 1].u)
        if inputs.ndim != 2 or inputs.shape[1] != players * 2:
            raise ValueError(f"iteration {number} has an inconsistent player count")
    costs = np.full((len(raw), players), np.nan)
    for index, iteration in enumerate(raw):
        for player in range(players):
            value = getattr(iteration, f"p{player + 1}_total_cost", None)
            if value is not None and np.isfinite(float(value)):
                costs[index, player] = float(value)
    return paths, costs, players, nx


def create_iteration_figure(learned_data, iterations, previous_alpha=0.12,
                            obstacle=(1.0, 0.0, 6.0, 2.0)):
    """Create a paper-sized row of trajectory panels.

    ``iterations`` contains one to four distinct, one-based iteration numbers,
    displayed in the supplied order. The figure opens no interactive window.
    ``obstacle`` is (center_x, center_y, semi_axis_x, semi_axis_y), or None.
    Defaults match the current Game.py; obstacle geometry is not in old pickles.
    """
    paths, costs, players, nx = _plot_data(learned_data)
    iterations = list(iterations)
    if not 1 <= len(iterations) <= 4 or len(set(iterations)) != len(iterations):
        raise ValueError("choose one to four distinct iterations")
    if any(not isinstance(i, (int, np.integer)) or not 1 <= i <= len(paths)
           for i in iterations):
        raise ValueError(f"iteration numbers must be between 1 and {len(paths)}")
    if not 0.0 <= previous_alpha <= 1.0:
        raise ValueError("previous_alpha must be between 0 and 1")
    if obstacle is not None:
        obstacle = np.asarray(obstacle, dtype=float)
        if (obstacle.shape != (4,) or not np.all(np.isfinite(obstacle))
                or np.any(obstacle[2:] <= 0)):
            raise ValueError("obstacle needs a finite center and two positive semi-axes")

    # Use common limits for the selected paths and their earlier histories.
    positions = np.concatenate([
        path[:, player * nx:player * nx + 2]
        for path in paths[:max(iterations)] for player in range(players)
    ])
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    if obstacle is not None:
        lower = np.minimum(lower, obstacle[:2] - obstacle[2:])
        upper = np.maximum(upper, obstacle[:2] + obstacle[2:])
    padding = np.maximum(0.08 * (upper - lower), 0.1)
    figure = Figure(figsize=(7.16, 1.9), layout="constrained")
    figure.get_layout_engine().set(w_pad=0.04, h_pad=0.06,
                                   wspace=0.025, hspace=0.04)
    grid = figure.add_gridspec(1, 4)
    for panel in range(4):
        ax = figure.add_subplot(grid[0, panel])
        if panel >= len(iterations):
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No iteration selected", transform=ax.transAxes,
                    ha="center", va="center", color="0.6")
            continue
        number = iterations[panel]
        current = paths[number - 1]
        times = np.asarray(learned_data.RawData[number - 1].t, dtype=float).reshape(-1)
        if (len(times) != len(current) or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0.0)):
            raise ValueError(f"iteration {number} needs increasing timestamps aligned with its states")
        marker_times = 0.5 * np.arange(np.floor(times[0] / 0.5) + 1,
                                       np.ceil(times[-1] / 0.5))
        marker_times = marker_times[(marker_times > times[0] + 1e-9)
                                    & (marker_times < times[-1] - 1e-9)]
        for player in range(players):
            offset, color = player * nx, f"C{player}"
            value = costs[number - 1, player]
            cost_label = f"{value:.0f}" if np.isfinite(value) else "n/a"
            for previous in paths[:number - 1]:
                ax.plot(previous[:, offset], previous[:, offset + 1], color=color,
                        linewidth=0.6, alpha=previous_alpha, zorder=1)
            ax.plot(current[:, offset], current[:, offset + 1], color=color,
                    linewidth=1.0, label=cost_label, zorder=3)
            # Interpolate on the saved path if a half-second falls between samples.
            marker_positions = np.column_stack([
                np.interp(marker_times, times, current[:, offset + axis])
                for axis in range(2)
            ])
            # Keep start/end locations reserved for their square/triangle markers,
            # including when a player waits at one of those locations.
            at_start = np.all(np.isclose(marker_positions, current[0, offset:offset + 2],
                                         rtol=0.0, atol=1e-9), axis=1)
            at_end = np.all(np.isclose(marker_positions, current[-1, offset:offset + 2],
                                       rtol=0.0, atol=1e-9), axis=1)
            marker_positions = marker_positions[~(at_start | at_end)]
            ax.plot(marker_positions[:, 0], marker_positions[:, 1], "o",
                    color=color, markersize=1.8, zorder=4)
            ax.plot(current[0, offset], current[0, offset + 1], "s",
                    color=color, markersize=3, zorder=4)
            ax.plot(current[-1, offset], current[-1, offset + 1], "^",
                    color=color, markersize=3, zorder=4)
        ax.set(xlim=(lower[0] - padding[0], upper[0] + padding[0]),
               ylim=(lower[1] - padding[1], upper[1] + padding[1]),
               xlabel="x position", ylabel="y position" if panel == 0 else "",
               title=f"Iteration {number}")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.tick_params(axis="y", labelleft=panel == 0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        if obstacle is not None:
            ax.add_patch(Ellipse(
                obstacle[:2], width=2 * obstacle[2], height=2 * obstacle[3],
                facecolor="0.85", edgecolor="0.3", linestyle="--",
                linewidth=0.6, hatch="///", zorder=0,
            ))
        ax.legend(loc="best", ncol=1, frameon=True, framealpha=0.85,
                  edgecolor="none", fontsize=6, handlelength=1.0,
                  handletextpad=0.3, labelspacing=0.2, borderaxespad=0.3)

    for ax in figure.axes:
        ax.tick_params(axis="both", which="both", labelsize=7, pad=2,
                       width=0.5, length=2)
        ax.title.set_fontsize(8)
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_fontsize(8)
            axis.labelpad = 2
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    return figure


def save_iteration_figure(
    learned_data, iterations, path="Results/iteration_comparison.png", previous_alpha=0.12,
    obstacle=(1.0, 0.0, 6.0, 2.0),
):
    """Save the selected-iteration figure as PNG, PDF, or SVG."""
    figure = create_iteration_figure(learned_data, iterations, previous_alpha, obstacle)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(output, dpi=600)
        if output.suffix.lower() == ".png":
            figure.savefig(output.with_suffix(".pdf"))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("./Results/Multiple_GNE_NashBargain.pkl"),
                        help="saved data file (default: Multiple_GNE_NashBargain_LearnedData.pkl)")
    parser.add_argument("--iterations", nargs="+", type=int, default=[1, 4, 8, 15],
                        help="up to four iteration numbers, starting at 1; defaults to evenly spaced iterations")
    parser.add_argument("--output", type=Path, default=Path("Results/iteration_comparison.png"))
    parser.add_argument("--previous-alpha", type=float, default=0.12)
    parser.add_argument("--obstacle", nargs=4, type=float,
                        metavar=("X", "Y", "A", "B"), default=(1.0, 0.0, 6.0, 2.0),
                        help="ellipse center and semi-axes (default: 1 0 6 2, matching Game.py)")
    parser.add_argument("--no-obstacle", action="store_true",
                        help="omit the obstacle for runs without one")
    args = parser.parse_args()
    try:
        learned_data = load_learned_data(args.data)
        paths, costs, _, _ = _plot_data(learned_data)
        iterations = args.iterations
        if iterations is None:
            iterations = list(np.linspace(1, len(paths), min(4, len(paths)), dtype=int))
        output = save_iteration_figure(
            learned_data, iterations, args.output, args.previous_alpha,
            None if args.no_obstacle else args.obstacle,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    plt.show()
    print(f"Saved {output}; selected iterations: {', '.join(map(str, iterations))}")
    if not np.all(np.isfinite(costs)):
        print("Some total costs are missing in the saved data; shown as n/a in legends.")


if __name__ == "__main__":
    main()
