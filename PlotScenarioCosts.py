"""Compare saved player costs across scenarios, with one subplot per player.

Examples:
    .venv/bin/python PlotScenarioCosts.py
    .venv/bin/python PlotScenarioCosts.py --no-show --output Results/costs.pdf

Loads every .pkl/.pickle file directly inside Results by default. Legend labels
omit the file extension and trailing _LearnedData. Iterations start at 1;
missing costs appear as gaps. The default figure is 3.5 inches wide for a single
column in a two-column paper, and is saved as both PNG and vector PDF.
Each player subplot includes its first-iteration cost as text.
Terminal summaries select the lowest sum of player costs at iteration 5 and
the final iteration, reporting savings relative to every other scenario.
"""

import argparse
from pathlib import Path
import pickle
import sys
import os
os.system('clear')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from LDG_Simulation_aux import load_learned_data


def load_scenario_costs(path):
    """Return an iteration-by-player array of saved total costs."""
    raw = load_learned_data(path).RawData
    if not raw:
        raise ValueError(f"{path}: no saved iterations")
    inputs = np.asarray(raw[0].u)
    if inputs.ndim != 2 or inputs.shape[1] not in (4, 6):
        raise ValueError(f"{path}: expected inputs for two or three players")
    players = inputs.shape[1] // 2
    costs = np.full((len(raw), players), np.nan)
    for index, iteration in enumerate(raw):
        inputs = np.asarray(iteration.u)
        if inputs.ndim != 2 or inputs.shape[1] != 2 * players:
            raise ValueError(f"{path}: inconsistent player count at iteration {index + 1}")
        for player in range(players):
            value = getattr(iteration, f"p{player + 1}_total_cost", None)
            if value is not None and np.isfinite(float(value)):
                costs[index, player] = float(value)
    return costs


def print_cost_summary(scenarios, iteration=None):
    """Report costs and savings at a one-based iteration, or the final one."""
    if iteration is not None and iteration < 1:
        raise ValueError("iteration must be at least 1")
    checkpoint = "Final iteration" if iteration is None else f"Iteration {iteration}"
    print(f"\n{checkpoint} cost comparison")
    if not scenarios:
        return
    if len({costs.shape[1] for costs in scenarios.values()}) != 1:
        print("Comparison unavailable: scenarios have different player counts.")
        return
    selected_costs = {}
    for name, costs in scenarios.items():
        if iteration is not None and len(costs) < iteration:
            print(f"Excluded {name}: iteration {iteration} is not available.")
            continue
        values = costs[-1 if iteration is None else iteration - 1]
        if np.isfinite(values).all():
            selected_costs[name] = values
        else:
            print(f"Excluded {name} from comparison: missing player costs.")
    if not selected_costs:
        print("No complete costs available for comparison.")
        return

    best_name = min(selected_costs, key=lambda name: selected_costs[name].sum())
    best = selected_costs[best_name]
    label = Path(best_name).stem.removesuffix("_LearnedData")
    print(f"Best scenario (lowest sum of player costs at this checkpoint): {label}")
    number = len(scenarios[best_name]) if iteration is None else iteration
    print(f"Iteration: {number}; total cost: {best.sum():.6f}")
    for player, value in enumerate(best, start=1):
        print(f"  Player {player} cost: {value:.6f}")
    print("\nSavings = other cost - best cost; positive means the best scenario is cheaper.")
    print("Savings (%) = 100 * savings / |other cost|; n/a when other cost is zero.")
    for name, other in selected_costs.items():
        if name == best_name:
            continue
        label = Path(name).stem.removesuffix("_LearnedData")
        number = len(scenarios[name]) if iteration is None else iteration
        print(f"\nCompared with {label} (iteration {number}):")
        print(f"{'Player':<10} {'Best cost':>14} {'Other cost':>14} {'Savings':>14} {'Savings (%)':>13}")
        for player, (best_value, other_value) in enumerate(zip(best, other), start=1):
            savings = other_value - best_value
            percent = f"{100 * savings / abs(other_value):+.2f}%" if other_value != 0 else "n/a"
            print(f"{player:<10} {best_value:>14.6f} {other_value:>14.6f} "
                  f"{savings:>+14.6f} {percent:>13}")


def create_scenario_figure(scenarios):
    """Plot a mapping of scenario labels to iteration-by-player cost arrays."""
    if not scenarios:
        raise ValueError("No scenarios to plot")
    players = max(costs.shape[1] for costs in scenarios.values())
    longest = max(len(costs) for costs in scenarios.values())
    height = 1.05 * players + 0.55
    figure, axes = plt.subplots(players, 1, sharex=True, squeeze=False,
                                figsize=(3.5, height))
    handles = []
    maxvals = np.zeros((players))
    for scenario_index, (label, costs) in enumerate(scenarios.items()):
        label = Path(label).stem.removesuffix("_LearnedData")
        iterations = np.arange(1, len(costs) + 1)
        for player in range(costs.shape[1]):
            line, = axes[player, 0].plot(
                iterations, costs[:, player], label=label,
                color=f"C{scenario_index % 10}", linewidth=1.0,
                marker=("o", "s", "^")[scenario_index % 3], markersize=2.4,
                markeredgewidth=0.4,
            )
            maxvals[player] = max(maxvals[player], 50+np.max(costs[1:,player]))
            if player == 0:
                handles.append(line)
    for player, ax in enumerate(axes[:, 0]):
        ax.set(ylabel=f"Player {player + 1} cost",
               xlim=(0.5, longest + 0.5))
        # ax.set_yscale("log", nonpositive="mask")
        ax.set_ylim([maxvals[player]-150, maxvals[player]])
        ax.yaxis.label.set_size(8)
        ax.yaxis.labelpad = 2
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.tick_params(axis="both", which="both", labelsize=7, pad=2,
                       width=0.5, length=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.grid(True, which="both", alpha=0.2, linewidth=0.4)
        first_costs = [
            (Path(name).stem.removesuffix("_LearnedData"), costs[0, player])
            for name, costs in scenarios.items() if player < costs.shape[1]
        ]
        first_labels = [
            f"{value:.2f}" if np.isfinite(value) else "n/a"
            for _, value in first_costs
        ]
        if len(set(first_labels)) == 1:
            first_text = f"Iteration 1: {first_labels[0]}"
        else:
            first_text = "Iteration 1:\n" + "\n".join(
                f"{name}: {value}"
                for (name, _), value in zip(first_costs, first_labels)
            )
        ax.text(0.03, 0.96, first_text, transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        if not any(player < costs.shape[1] and np.isfinite(costs[:, player]).any()
                   for costs in scenarios.values()):
            ax.text(0.5, 0.5, "No saved total costs", transform=ax.transAxes,
                    ha="center", va="center")
    axes[-1, 0].set_xlabel("Iterations", fontsize=8, labelpad=2)
    axes[0, 0].legend(handles=handles, loc="upper right",
                      frameon=False, fontsize=7, handlelength=1.4,
                      handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3)
    figure.align_ylabels(axes[:, 0])
    figure.tight_layout(pad=0.5, h_pad=0.35)
    return figure


def main():
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="", flush=True)
    default_results = Path(__file__).resolve().parent / "Results"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=default_results,
                        help="directory containing scenario pickle files (default: Results)")
    parser.add_argument("--output", type=Path,
                        help="output image path (default: RESULTS_DIR/scenario_costs.png)")
    parser.add_argument("--no-show", action="store_true",
                        help="save the figure without opening a window")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    paths = sorted(path for path in args.results_dir.glob("*")
                   if path.is_file() and path.suffix.lower() in (".pkl", ".pickle"))
    if not paths:
        parser.error(f"No pickle files found in {args.results_dir}")
    scenarios = {}
    try:
        for path in paths:
            costs = load_scenario_costs(path)
            scenarios[path.name] = costs
            print(f"{path.name}: {len(costs)} iterations, {costs.shape[1]} players")
            if not np.isfinite(costs).all():
                print("  Missing/nonfinite costs are shown as gaps.")
        print_cost_summary(scenarios, iteration=5)
        print_cost_summary(scenarios)
        figure = create_scenario_figure(scenarios)
        output = args.output or args.results_dir / "scenario_costs.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        # Keep the exact column width; tight_layout already fits the labels.
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            figure.savefig(output, dpi=600)
            if output.suffix.lower() == ".png":
                pdf_output = output.with_suffix(".pdf")
                figure.savefig(pdf_output)
                print(f"Saved {pdf_output}")
    except (OSError, ValueError, EOFError, pickle.UnpicklingError) as error:
        parser.error(str(error))
    print(f"Saved {output}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
