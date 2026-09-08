"""Plot selected learning iterations from a saved LearnedData.pkl file.

Example:
    .venv/bin/python PlotLearnedIterations.py --iterations 1 3 5 10

Iteration numbers start at 1. Each panel includes all earlier paths with low
opacity. The bottom plot uses saved total costs across every iteration; missing
costs are shown as n/a in panels and as gaps in the cost plot.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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


def create_iteration_figure(learned_data, iterations, previous_alpha=0.12):
    """Create a 2x2 XY comparison plus a compact plot of all saved costs.

    ``iterations`` contains one to four distinct, one-based iteration numbers,
    displayed in the supplied order. The figure opens no interactive window.
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

    # Use common limits for the selected paths and their earlier histories.
    positions = np.concatenate([
        path[:, player * nx:player * nx + 2]
        for path in paths[:max(iterations)] for player in range(players)
    ])
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    padding = np.maximum(0.08 * (upper - lower), 0.1)
    figure = Figure(figsize=(12, 10), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.25))
    for panel in range(4):
        cell = grid[panel // 2, panel % 2].subgridspec(
            2, 1, height_ratios=(1.0, 0.05), hspace=0.02,
        )
        ax = figure.add_subplot(cell[0])
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
            for previous in paths[:number - 1]:
                ax.plot(previous[:, offset], previous[:, offset + 1], color=color,
                        linewidth=1.0, alpha=previous_alpha, zorder=1)
            ax.plot(current[:, offset], current[:, offset + 1], color=color,
                    linewidth=2.0, label=f"P{player + 1}", zorder=3)
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
                    color=color, markersize=4, zorder=4)
            ax.plot(current[0, offset], current[0, offset + 1], "s",
                    color=color, markersize=6, zorder=4)
            ax.plot(current[-1, offset], current[-1, offset + 1], "^",
                    color=color, markersize=6, zorder=4)
        ax.set(xlim=(lower[0] - padding[0], upper[0] + padding[0]),
               ylim=(lower[1] - padding[1], upper[1] + padding[1]),
               xlabel="x position", ylabel="y position", title=f"Iteration {number}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        if panel == 0:
            ax.legend(loc="best", ncol=players, frameon=False, fontsize=8)
        cost_text = "   ".join(
            f"P{player + 1}: {value:.2f}" if np.isfinite(value) else f"P{player + 1}: n/a"
            for player, value in enumerate(costs[number - 1])
        )
        # footer.text(0.98, 0.5, f"Current cost   {cost_text}",
        #             transform=footer.transAxes, ha="right", va="center", fontsize=9)

    ax_cost = figure.add_subplot(grid[2, :])
    numbers = np.arange(1, len(paths) + 1)
    for player in range(players):
        ax_cost.plot(numbers, costs[:, player], "-o", color=f"C{player}",
                     linewidth=1.5, markersize=4, label=f"P{player + 1}", zorder=2)
    for number in iterations:
        ax_cost.axvline(number, color="0.7", linewidth=0.8, alpha=0.4, zorder=0)
    ax_cost.set(xlabel="iteration", ylabel="total cost", title="Player costs across iterations",
                xlim=(0.5, len(paths) + 0.5))
    ax_cost.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_cost.grid(True, alpha=0.25)
    ax_cost.legend(loc="best", ncol=players, frameon=False, fontsize=8)
    if not np.any(np.isfinite(costs)):
        ax_cost.text(0.5, 0.5, "No total costs stored in this file",
                     transform=ax_cost.transAxes, ha="center", va="center", color="0.4")
    return figure


def save_iteration_figure(
    learned_data, iterations, path="Results/iteration_comparison.png", previous_alpha=0.12,
):
    """Save the selected-iteration figure as PNG, PDF, or SVG."""
    figure = create_iteration_figure(learned_data, iterations, previous_alpha)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("./Results/LearnedData.pkl"),
                        help="saved data file (default: LearnedData.pkl)")
    parser.add_argument("--iterations", nargs="+", type=int, default=[1, 4, 8, 15],
                        help="up to four iteration numbers, starting at 1; defaults to evenly spaced iterations")
    parser.add_argument("--output", type=Path, default=Path("Results/iteration_comparison.png"))
    parser.add_argument("--previous-alpha", type=float, default=0.12)
    args = parser.parse_args()
    try:
        learned_data = load_learned_data(args.data)
        paths, costs, _, _ = _plot_data(learned_data)
        iterations = args.iterations
        if iterations is None:
            iterations = list(np.linspace(1, len(paths), min(4, len(paths)), dtype=int))
        output = save_iteration_figure(learned_data, iterations, args.output, args.previous_alpha)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    plt.show()
    print(f"Saved {output}; selected iterations: {', '.join(map(str, iterations))}")
    if not np.all(np.isfinite(costs)):
        print("Some total costs are missing in the saved data; shown as n/a and gaps.")


if __name__ == "__main__":
    main()
