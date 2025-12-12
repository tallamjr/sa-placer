#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "polars>=1.0",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
# ]
# ///
"""
Generate xkcd-style visualisations for the SA-Placer blog post.

Uses matplotlib's xkcd mode for a hand-drawn, engaging aesthetic
that fits the informal blog post tone.

Usage:
    uv run scripts/generate_plots.py
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

OUTPUT_DIR = Path(__file__).parent.parent / 'output_data'
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Colour palette - muted, professional
BLUE = '#4878A8'
CORAL = '#E57A5A'
GREEN = '#5A9E5A'
PURPLE = '#8B6BB8'
ORANGE = '#D69A2E'
GREY = '#666666'


def load_greedy_data() -> dict[int, pl.DataFrame]:
    """Load all greedy descent CSV files from main.rs output."""
    data = {}
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        csv_path = OUTPUT_DIR / f'fpga_placer_history_{n}.csv'
        if csv_path.exists():
            data[n] = pl.read_csv(csv_path)
    return data


def load_comparison_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None:
    """Load fair budget comparison data if available."""
    comparison_dir = OUTPUT_DIR / 'comparison'
    greedy_path = comparison_dir / 'greedy_convergence.csv'
    sa_path = comparison_dir / 'sa_convergence.csv'
    summary_path = comparison_dir / 'summary.csv'

    if greedy_path.exists() and sa_path.exists() and summary_path.exists():
        return (
            pl.read_csv(greedy_path),
            pl.read_csv(sa_path),
            pl.read_csv(summary_path)
        )
    return None


def plot_convergence_all_neighbors_xkcd(data: dict[int, pl.DataFrame]) -> None:
    """Recreate fpga_placer_history.png in xkcd style."""
    with plt.xkcd(scale=1, length=100, randomness=2):
        fig, ax = plt.subplots(figsize=(14, 8))

        # Colour cycle for different n_neighbors
        colors = [PURPLE, '#2E8B57', '#20B2AA', GREY, ORANGE,
                  GREEN, CORAL, 'black', '#FF69B4']

        for i, (n, df) in enumerate(sorted(data.items())):
            steps = df['step'].to_numpy()
            costs = df['obj_fn_value'].to_numpy()
            color = colors[i % len(colors)]
            ax.plot(steps, costs, linewidth=2, label=f'{n} neighbours', color=color)

        ax.set_xlabel('step', fontsize=14)
        ax.set_ylabel('objective function value', fontsize=14)
        ax.set_title('greedy descent convergence: all configurations\n(monotonically decreasing = no uphill moves)',
                     fontsize=16, pad=15)
        ax.legend(fontsize=11, loc='upper right', ncol=2)
        ax.set_xlim(0, 1000)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add annotation about monotonicity
        ax.annotate('all curves only go DOWN\n(pure greedy behaviour)',
                    xy=(600, 45000), fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.5))

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'convergence_all_neighbors.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'convergence_all_neighbors.png'}")


def plot_final_cost_vs_neighbors_xkcd(data: dict[int, pl.DataFrame]) -> None:
    """Bar chart with xkcd styling."""
    with plt.xkcd(scale=1, length=100, randomness=2):
        neighbors = []
        final_costs = []

        for n, df in sorted(data.items()):
            neighbors.append(n)
            final_costs.append(df['obj_fn_value'][-1])

        fig, ax = plt.subplots(figsize=(12, 7))

        bars = ax.bar(range(len(neighbors)), final_costs, color=BLUE,
                      edgecolor='black', linewidth=2, alpha=0.85)

        # Highlight optimal
        min_idx = np.argmin(final_costs)
        bars[min_idx].set_color(GREEN)

        ax.set_xticks(range(len(neighbors)))
        ax.set_xticklabels(neighbors, fontsize=14)
        ax.set_xlabel('neighbours evaluated per step', fontsize=16)
        ax.set_ylabel('final cost', fontsize=16)
        ax.set_title('more neighbours helps... to a point', fontsize=20, pad=20)

        # Value labels
        for bar, cost in zip(bars, final_costs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                    f'{int(cost):,}', ha='center', va='bottom', fontsize=11)

        # Annotation
        ax.annotate('sweet spot!',
                    xy=(min_idx, final_costs[min_idx]),
                    xytext=(min_idx + 1.5, final_costs[min_idx] + 4000),
                    fontsize=14, color=GREEN,
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'final_cost_vs_neighbors.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'final_cost_vs_neighbors.png'}")


def plot_greedy_vs_sa_fair(greedy_df: pl.DataFrame, sa_df: pl.DataFrame,
                           summary_df: pl.DataFrame) -> None:
    """Plot fair budget comparison data from experiments.

    Both algorithms have equal candidate evaluations, so the x-axis
    represents equivalent computational effort.
    """
    with plt.xkcd(scale=1, length=100, randomness=2):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Get seed columns
        seed_cols = [c for c in greedy_df.columns if c.startswith('seed_')]
        n_seeds = len(seed_cols)

        # Left: Full convergence with mean and confidence bands
        ax1 = axes[0]

        # X-axis is candidate evaluations (same for both algorithms)
        x_evals = greedy_df['candidate_evaluations'].to_numpy()

        # Compute mean and std for greedy
        greedy_data = greedy_df.select(seed_cols).to_numpy()
        greedy_mean = np.mean(greedy_data, axis=1)
        greedy_std = np.std(greedy_data, axis=1)

        # Compute mean and std for SA
        sa_data = sa_df.select(seed_cols).to_numpy()
        sa_mean = np.mean(sa_data, axis=1)
        sa_std = np.std(sa_data, axis=1)

        # Plot with confidence bands
        ax1.fill_between(x_evals, greedy_mean - greedy_std, greedy_mean + greedy_std,
                         alpha=0.3, color=BLUE)
        ax1.plot(x_evals, greedy_mean, color=BLUE, linewidth=2.5, label='greedy descent')

        ax1.fill_between(x_evals, sa_mean - sa_std, sa_mean + sa_std,
                         alpha=0.3, color=CORAL)
        ax1.plot(x_evals, sa_mean, color=CORAL, linewidth=2.5, label='true SA')

        ax1.set_xlabel('candidate evaluations', fontsize=14)
        ax1.set_ylabel('cost', fontsize=14)
        ax1.set_title('fair budget comparison\n(equal computational effort, shaded = 1 std dev)',
                      fontsize=16, pad=10)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Right: Zoomed view showing SA non-monotonicity
        ax2 = axes[1]
        # Zoom to middle section where behaviour differences are visible
        zoom_start, zoom_end = 50, 300
        if len(x_evals) > zoom_end:
            x_zoom = x_evals[zoom_start:zoom_end]

            # Plot individual SA trajectories to show wiggles
            for i, col in enumerate(seed_cols[:3]):  # Show 3 seeds
                sa_vals = sa_df[col].to_numpy()[zoom_start:zoom_end]
                alpha = 0.6 if i > 0 else 0.9
                label = 'SA trajectories' if i == 0 else None
                ax2.plot(x_zoom, sa_vals,
                         color=CORAL, alpha=alpha, linewidth=1.5, label=label)

            # Show greedy mean for comparison
            ax2.plot(x_zoom, greedy_mean[zoom_start:zoom_end],
                     color=BLUE, linewidth=2.5, label='greedy (mean)')

            # Find and annotate an uphill move in SA
            for col in seed_cols[:1]:
                sa_vals = sa_df[col].to_numpy()[zoom_start:zoom_end]
                for j in range(1, len(sa_vals)):
                    if sa_vals[j] > sa_vals[j-1] + 50:  # Significant uphill
                        ax2.annotate('uphill move!',
                                    xy=(x_zoom[j], sa_vals[j]),
                                    xytext=(x_zoom[j] + 500, sa_vals[j] + 2000),
                                    fontsize=12, color='red',
                                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
                        break

            ax2.set_xlabel('candidate evaluations', fontsize=14)
            ax2.set_ylabel('cost', fontsize=14)
            ax2.set_title('zoomed: SA accepts worse solutions\n(greedy never does)',
                          fontsize=16, pad=10)
            ax2.legend(fontsize=11, loc='upper right')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'greedy_vs_sa_behaviour.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'greedy_vs_sa_behaviour.png'}")


def plot_monotonicity_evidence_xkcd() -> None:
    """Monotonicity proof with xkcd style."""
    df = pl.read_csv(OUTPUT_DIR / 'fpga_placer_history_16.csv')
    costs = df['obj_fn_value'].to_numpy()
    steps = df['step'].to_numpy()
    deltas = np.diff(costs)

    with plt.xkcd(scale=1, length=100, randomness=2):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1])

        # Top: cost trajectory
        ax1.plot(steps, costs, color=BLUE, linewidth=2.5)
        ax1.fill_between(steps, costs, alpha=0.3, color=BLUE)
        ax1.set_ylabel('cost', fontsize=14)
        ax1.set_title('greedy descent: always going down\n(n_neighbors=16, 1000 iterations)',
                      fontsize=18, pad=15)
        ax1.set_xlim(0, 1000)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Bottom: deltas
        colors = [GREEN if d <= 0 else 'red' for d in deltas]
        ax2.bar(range(len(deltas)), deltas, color=colors, width=1.0, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('iteration', fontsize=14)
        ax2.set_ylabel('cost change', fontsize=14)
        ax2.set_title('step-by-step: never positive!', fontsize=16)
        ax2.set_xlim(0, 1000)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Stats
        positive = int(np.sum(deltas > 0))
        negative = int(np.sum(deltas < 0))
        zero = int(np.sum(deltas == 0))

        stats_text = f'improvements: {negative}\nno change: {zero}\nuphill: {positive}'
        ax2.text(0.97, 0.95, stats_text, transform=ax2.transAxes,
                 fontsize=13, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.5))

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'monotonicity_evidence.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'monotonicity_evidence.png'}")


def plot_cost_landscape_3d() -> None:
    """3D surface plot showing local minima landscape."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 7))

    # Left: 2D cross-section
    ax1 = fig.add_subplot(121)
    x = np.linspace(0, 10, 500)
    y = (np.sin(x * 2) * 2 + np.sin(x * 0.5) * 4 + 10 -
         0.3 * x + np.sin(x * 5) * 0.5)

    with plt.xkcd(scale=1, length=100, randomness=2):
        ax1.plot(x, y, 'k-', linewidth=2.5)
        ax1.fill_between(x, y, alpha=0.2, color=BLUE)

        # Local and global minima
        local_min_x = [1.5, 4.7]
        local_min_y = [y[int(lm * 50)] for lm in local_min_x]
        global_min_x = 8.5
        global_min_y = y[int(global_min_x * 50)]

        ax1.scatter(local_min_x, local_min_y, s=200, c=ORANGE, zorder=5,
                   label='local minima', edgecolor='black', linewidth=2)
        ax1.scatter([global_min_x], [global_min_y], s=350, c=GREEN, zorder=5,
                   label='global minimum', edgecolor='black', linewidth=2, marker='*')

        # Repositioned annotations to avoid title overlap
        ax1.annotate('greedy gets\nstuck here',
                    xy=(local_min_x[0], local_min_y[0]),
                    xytext=(local_min_x[0] + 1.8, local_min_y[0] + 1.0),
                    fontsize=13, color=ORANGE, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=ORANGE, alpha=0.9))

        ax1.annotate('SA might\nescape!',
                    xy=(3.0, y[150]),
                    xytext=(4.8, y[100] - 1.5),
                    fontsize=13, color=CORAL, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=CORAL, lw=2.5,
                                   connectionstyle='arc3,rad=0.2'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=CORAL, alpha=0.9))

        ax1.set_xlabel('solution space', fontsize=14)
        ax1.set_ylabel('cost', fontsize=14)
        ax1.set_title('cost landscape (1D slice)', fontsize=18, pad=15)
        ax1.legend(loc='upper right', fontsize=12)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(y.min() - 1, y.max() + 2)  # Dynamic limits to show all points
        ax1.set_yticklabels([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

    # Right: 3D surface
    ax2 = fig.add_subplot(122, projection='3d')

    # Create a 2D cost landscape with multiple local minima
    x_3d = np.linspace(-3, 3, 100)
    y_3d = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_3d, y_3d)

    # Rastrigin-like function with local minima
    Z = (X**2 + Y**2) + 2 * (1 - np.cos(2*np.pi*X)) + 2 * (1 - np.cos(2*np.pi*Y))
    Z = Z / Z.max() * 20 + 5  # Scale to reasonable range

    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.75,
                           linewidth=0, antialiased=True)

    # Mark global minimum with larger marker
    ax2.scatter([0], [0], [Z.min() + 0.5], color=GREEN, s=400, marker='*',
               edgecolor='black', linewidth=2, zorder=10, label='global minimum')

    # Mark some local minima with larger markers
    local_x = [1, -1, 1, -1]
    local_y = [1, 1, -1, -1]
    for i, (lx, ly) in enumerate(zip(local_x, local_y)):
        idx_x = int((lx + 3) / 6 * 99)
        idx_y = int((ly + 3) / 6 * 99)
        label = 'local minima' if i == 0 else None
        ax2.scatter([lx], [ly], [Z[idx_y, idx_x] + 0.5], color=ORANGE, s=180,
                   marker='o', edgecolor='black', linewidth=2, zorder=10, label=label)

    ax2.set_xlabel('x', fontsize=13, labelpad=8)
    ax2.set_ylabel('y', fontsize=13, labelpad=8)
    ax2.set_zlabel('cost', fontsize=13, labelpad=8)
    ax2.set_title('3D cost landscape\n(many local minima!)', fontsize=18, pad=10)
    ax2.view_init(elev=30, azim=45)
    ax2.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cost_landscape_intuition.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'cost_landscape_intuition.png'}")


def plot_cooling_schedules_xkcd() -> None:
    """Cooling schedules with xkcd style."""
    with plt.xkcd(scale=1, length=100, randomness=2):
        fig, ax = plt.subplots(figsize=(12, 7))

        steps = np.arange(0, 1000)
        T0 = 5000

        # Geometric
        geometric = T0 * (0.995 ** steps)

        # Linear
        linear = np.maximum(T0 - 5 * steps, 0)

        # Logarithmic
        logarithmic = T0 / np.log(steps + 2)

        # Adaptive (simulated)
        adaptive = [T0]
        temp = T0
        for i in range(1, len(steps)):
            acceptance_rate = 0.8 * np.exp(-i / 300)
            if acceptance_rate > 0.4:
                temp *= 0.99
            else:
                temp *= 0.995
            adaptive.append(temp)

        ax.plot(steps, geometric, label='geometric', linewidth=3, color=BLUE)
        ax.plot(steps, linear, label='linear', linewidth=3, color=CORAL)
        ax.plot(steps, logarithmic, label='logarithmic (slow!)', linewidth=3, color=GREEN)
        ax.plot(steps, adaptive, label='adaptive', linewidth=3, color=PURPLE, linestyle='--')

        ax.set_xlabel('iteration', fontsize=14)
        ax.set_ylabel('temperature', fontsize=14)
        ax.set_title('cooling schedules: how fast to freeze?', fontsize=18, pad=15)
        ax.legend(fontsize=12, loc='upper right')
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 5500)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotation
        ax.annotate('hot = explore freely\ncold = be greedy',
                    xy=(500, 2500), fontsize=13,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.5))

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'cooling_schedules.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'cooling_schedules.png'}")


def plot_experiment_results(summary_df: pl.DataFrame) -> None:
    """Plot actual experiment results from fair budget comparison."""
    with plt.xkcd(scale=1, length=100, randomness=2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        seeds = summary_df['seed'].to_numpy()
        greedy = summary_df['greedy_final'].to_numpy()
        sa_best = summary_df['sa_best'].to_numpy()  # Use best, not final
        uphill = summary_df['uphill'].to_numpy()

        # Left: bar comparison
        x = np.arange(len(seeds))
        width = 0.35

        bars1 = ax1.bar(x - width/2, greedy, width, label='greedy', color=BLUE, alpha=0.85)
        bars2 = ax1.bar(x + width/2, sa_best, width, label='true SA (best)', color=CORAL, alpha=0.85)

        ax1.set_xlabel('random seed', fontsize=14)
        ax1.set_ylabel('final cost', fontsize=14)
        ax1.set_title('fair budget comparison: greedy vs SA\n(16,000 candidate evaluations each)',
                      fontsize=16)
        ax1.set_xticks(x)
        ax1.legend(fontsize=12)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Right: uphill moves
        ax2.bar(seeds, uphill, color=CORAL, alpha=0.85, edgecolor='black')
        ax2.set_xlabel('random seed', fontsize=14)
        ax2.set_ylabel('uphill moves accepted', fontsize=14)
        ax2.set_title('SA really does accept uphill moves!', fontsize=16)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Average annotation
        avg_uphill = np.mean(uphill)
        ax2.axhline(y=avg_uphill, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.annotate(f'avg: {avg_uphill:.0f}',
                    xy=(len(seeds)-1, avg_uphill),
                    xytext=(len(seeds)-0.5, avg_uphill + 50),
                    fontsize=12, color='red')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'experiment_results.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'experiment_results.png'}")


def print_verification_summary(summary_df: pl.DataFrame) -> None:
    """Print summary statistics for README verification."""
    greedy = summary_df['greedy_final'].to_numpy()
    sa_best = summary_df['sa_best'].to_numpy()
    uphill = summary_df['uphill'].to_numpy()
    initial = summary_df['initial'].to_numpy()

    avg_initial = np.mean(initial)
    avg_greedy = np.mean(greedy)
    avg_sa_best = np.mean(sa_best)
    avg_uphill = np.mean(uphill)

    greedy_reduction = (1 - avg_greedy / avg_initial) * 100
    sa_reduction = (1 - avg_sa_best / avg_initial) * 100

    print("\n" + "=" * 60)
    print("VERIFICATION: These values should match README.md")
    print("=" * 60)
    print(f"  Average initial cost:   {avg_initial:.0f}")
    print(f"  Average greedy final:   {avg_greedy:.0f} ({greedy_reduction:.1f}% reduction)")
    print(f"  Average SA best:        {avg_sa_best:.0f} ({sa_reduction:.1f}% reduction)")
    print(f"  Average uphill moves:   {avg_uphill:.0f}")

    if avg_greedy < avg_sa_best:
        winner_pct = (1 - avg_greedy / avg_sa_best) * 100
        print(f"\n  Greedy wins by: {winner_pct:.1f}%")
    else:
        winner_pct = (1 - avg_sa_best / avg_greedy) * 100
        print(f"\n  SA wins by: {winner_pct:.1f}%")
    print("=" * 60 + "\n")


def main() -> None:
    print("Generating xkcd-style SA-Placer visualisations...")
    print(f"Output directory: {FIGURES_DIR}")
    print()

    # Load greedy sweep data (from main.rs)
    data = load_greedy_data()
    if data:
        print(f"Loaded greedy sweep data for n_neighbors: {list(data.keys())}")
    else:
        print("Warning: No greedy sweep data found in output_data/")
        print("Run 'cargo run --release' first to generate this data.")

    # Load comparison data (from compare.rs)
    comparison_data = load_comparison_data()

    if not data and not comparison_data:
        print("ERROR: No data found. Run experiments first:")
        print("  cargo run --release           # greedy sweep")
        print("  cargo run --release --bin compare  # fair comparison")
        return

    # Generate plots that use greedy sweep data
    if data:
        plot_convergence_all_neighbors_xkcd(data)
        plot_final_cost_vs_neighbors_xkcd(data)
        plot_monotonicity_evidence_xkcd()

    # Always generate these (they don't need data)
    plot_cost_landscape_3d()
    plot_cooling_schedules_xkcd()

    # Generate plots that use comparison data
    if comparison_data:
        greedy_df, sa_df, summary_df = comparison_data
        print("\nUsing REAL fair-budget comparison data!")
        plot_greedy_vs_sa_fair(greedy_df, sa_df, summary_df)
        plot_experiment_results(summary_df)
        print_verification_summary(summary_df)
    else:
        print("\nNo comparison data found.")
        print("Run 'cargo run --release --bin compare' to generate comparison data.")

    print()
    print("All plots generated!")
    print(f"View plots in: {FIGURES_DIR.absolute()}")


if __name__ == '__main__':
    main()
