#!/usr/bin/env python3
"""
Generate Phase 4 visualizations.

Creates charts specific to Phase 4 questions:
1. Extended depth collapse curves (sigmoid heading to 50%)
2. Skip connection rescue effect
3. Sine stability at extreme depths
4. Paired baseline vs skip comparison
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import load_from_csv, ConfigurationStats
from typing import List, Dict


# Color schemes - include both full names and 3-letter codes
ACTIVATION_COLORS = {
    'relu': '#3498db',
    'rel': '#3498db',
    'tanh': '#2ecc71',
    'tan': '#2ecc71',
    'gelu': '#f39c12',
    'gel': '#f39c12',
    'sigmoid': '#e74c3c',
    'sig': '#e74c3c',
    'sine': '#9b59b6',
    'sin': '#9b59b6',
    'leaky_relu': '#1abc9c',
    'lea': '#1abc9c',
}

ACTIVATION_LABELS = {
    'relu': 'ReLU',
    'rel': 'ReLU',
    'tanh': 'Tanh',
    'tan': 'Tanh',
    'gelu': 'GELU',
    'gel': 'GELU',
    'sigmoid': 'Sigmoid',
    'sig': 'Sigmoid',
    'sine': 'Sine',
    'sin': 'Sine',
    'leaky_relu': 'Leaky ReLU',
    'lea': 'Leaky ReLU',
}

# Canonical activation names for iteration
PHASE4_ACTIVATIONS = ['sigmoid', 'sig', 'relu', 'rel', 'sine', 'sin', 'tanh', 'tan']


def create_extended_depth_collapse(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Extended Depth Study: Activation Degradation"
) -> None:
    """
    Show how each activation degrades with depth from 1-10.
    Highlights sigmoid's collapse toward 50% (random chance).
    """
    # Filter to baseline only (no skip connections)
    baseline = [s for s in stats if not s.skip_connections]

    # Group by activation and depth, averaging across datasets
    by_activation: Dict[str, Dict[int, List]] = {}
    for s in baseline:
        act = s.activation
        if act not in by_activation:
            by_activation[act] = {}
        if s.depth not in by_activation[act]:
            by_activation[act][s.depth] = []
        by_activation[act][s.depth].append(s)

    fig, ax = plt.subplots(figsize=(12, 7))

    all_depths = sorted(set(s.depth for s in baseline))

    for act in PHASE4_ACTIVATIONS:
        if act not in by_activation:
            continue

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())

        depths = []
        means = []
        ci_lowers = []
        ci_uppers = []

        for d in sorted(by_activation[act].keys()):
            stat_list = by_activation[act][d]
            all_means = [s.mean_accuracy for s in stat_list]
            mean_acc = np.mean(all_means)
            ci_lower = mean_acc - np.mean([s.mean_accuracy - s.ci_lower for s in stat_list])
            ci_upper = mean_acc + np.mean([s.ci_upper - s.mean_accuracy for s in stat_list])

            depths.append(d)
            means.append(mean_acc)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)

        # Plot shaded CI
        ax.fill_between(depths, ci_lowers, ci_uppers, color=color, alpha=0.2)
        # Plot mean line
        line_style = '-' if act != 'sigmoid' else '--'
        line_width = 2 if act != 'sigmoid' else 3
        ax.plot(depths, means, f'o{line_style}', color=color, linewidth=line_width,
               markersize=8, label=label)

        # Annotate sigmoid at depth 10
        if act == 'sigmoid' and 10 in by_activation[act]:
            final_acc = means[-1] if depths[-1] == 10 else None
            if final_acc:
                ax.annotate(f'{final_acc*100:.1f}%',
                           xy=(depths[-1], final_acc),
                           xytext=(10, -30),
                           textcoords='offset points',
                           fontsize=11,
                           fontweight='bold',
                           color=color,
                           arrowprops=dict(arrowstyle='->', color=color))

    # Random chance line
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Random chance (50%)')

    ax.set_xlabel('Network Depth (hidden layers)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(all_depths)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_skip_connection_rescue(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Skip Connection Rescue Effect"
) -> None:
    """
    Paired comparison showing baseline vs skip connections for each activation.
    Focus on whether skip connections rescue deep sigmoid networks.
    """
    baseline = [s for s in stats if not s.skip_connections]
    with_skip = [s for s in stats if s.skip_connections]

    if not with_skip:
        print("No skip connection experiments found - skipping skip_connection_rescue plot")
        return

    # Focus on Phase 4 depths
    phase4_depths = {6, 7, 8, 10}
    baseline_p4 = [s for s in baseline if s.depth in phase4_depths]
    with_skip_p4 = [s for s in with_skip if s.depth in phase4_depths]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Get unique activations from data (handles both full names and short codes)
    available_activations = sorted(set(s.activation for s in baseline_p4))
    activations = available_activations[:4]  # Take up to 4 for the 2x2 grid

    for ax, act in zip(axes.flat, activations):
        base_act = [s for s in baseline_p4 if s.activation == act]
        skip_act = [s for s in with_skip_p4 if s.activation == act]

        # Group by depth
        base_by_depth = {}
        skip_by_depth = {}

        for s in base_act:
            if s.depth not in base_by_depth:
                base_by_depth[s.depth] = []
            base_by_depth[s.depth].append(s.mean_accuracy)

        for s in skip_act:
            if s.depth not in skip_by_depth:
                skip_by_depth[s.depth] = []
            skip_by_depth[s.depth].append(s.mean_accuracy)

        depths = sorted(set(base_by_depth.keys()) | set(skip_by_depth.keys()))
        x = np.arange(len(depths))
        width = 0.35

        base_means = [np.mean(base_by_depth.get(d, [0])) for d in depths]
        skip_means = [np.mean(skip_by_depth.get(d, [0])) for d in depths]

        color = ACTIVATION_COLORS.get(act, '#888888')

        bars1 = ax.bar(x - width/2, base_means, width, label='Baseline',
                      color=color, alpha=0.7)
        bars2 = ax.bar(x + width/2, skip_means, width, label='With Skip',
                      color=color, edgecolor='black', linewidth=2)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{ACTIVATION_LABELS.get(act, act.upper())}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'd={d}' for d in depths])
        ax.set_ylim(0.4, 1.05)
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate improvement
        for i, d in enumerate(depths):
            if base_means[i] > 0 and skip_means[i] > 0:
                delta = (skip_means[i] - base_means[i]) * 100
                if abs(delta) > 1:  # Only show if > 1% difference
                    symbol = '+' if delta > 0 else ''
                    color_txt = 'green' if delta > 0 else 'red'
                    ax.text(i, max(base_means[i], skip_means[i]) + 0.02,
                           f'{symbol}{delta:.0f}%', ha='center', fontsize=9,
                           fontweight='bold', color=color_txt)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_sigmoid_collapse_curve(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Sigmoid Collapse: Approaching Random Chance"
) -> None:
    """
    Detailed view of sigmoid's degradation with depth.
    Shows both baseline and skip connection variants.
    """
    # Match both 'sigmoid' and 'sig' (3-letter code)
    sigmoid_baseline = [s for s in stats if s.activation in ('sigmoid', 'sig') and not s.skip_connections]
    sigmoid_skip = [s for s in stats if s.activation in ('sigmoid', 'sig') and s.skip_connections]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by depth
    def aggregate_by_depth(stat_list):
        by_depth = {}
        for s in stat_list:
            if s.depth not in by_depth:
                by_depth[s.depth] = []
            by_depth[s.depth].append(s)
        return by_depth

    base_by_depth = aggregate_by_depth(sigmoid_baseline)
    skip_by_depth = aggregate_by_depth(sigmoid_skip)

    # Plot baseline
    if base_by_depth:
        depths = sorted(base_by_depth.keys())
        means = []
        ci_lowers = []
        ci_uppers = []

        for d in depths:
            accs = [s.mean_accuracy for s in base_by_depth[d]]
            mean_acc = np.mean(accs)
            ci_width = np.mean([s.ci_upper - s.ci_lower for s in base_by_depth[d]]) / 2
            means.append(mean_acc)
            ci_lowers.append(mean_acc - ci_width)
            ci_uppers.append(mean_acc + ci_width)

        ax.fill_between(depths, ci_lowers, ci_uppers, color='#e74c3c', alpha=0.2)
        ax.plot(depths, means, 'o-', color='#e74c3c', linewidth=2.5,
               markersize=10, label='Sigmoid (baseline)')

        # Add percentage labels
        for d, m in zip(depths, means):
            ax.annotate(f'{m*100:.1f}%', xy=(d, m), xytext=(0, 10),
                       textcoords='offset points', ha='center', fontsize=9)

    # Plot with skip connections
    if skip_by_depth:
        depths = sorted(skip_by_depth.keys())
        means = []
        ci_lowers = []
        ci_uppers = []

        for d in depths:
            accs = [s.mean_accuracy for s in skip_by_depth[d]]
            mean_acc = np.mean(accs)
            ci_width = np.mean([s.ci_upper - s.ci_lower for s in skip_by_depth[d]]) / 2
            means.append(mean_acc)
            ci_lowers.append(mean_acc - ci_width)
            ci_uppers.append(mean_acc + ci_width)

        ax.fill_between(depths, ci_lowers, ci_uppers, color='#27ae60', alpha=0.2)
        ax.plot(depths, means, 's--', color='#27ae60', linewidth=2.5,
               markersize=10, label='Sigmoid (with skip)')

    # Random chance line
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(1.5, 0.51, 'Random chance (50%)', fontsize=10, color='gray')

    ax.set_xlabel('Network Depth (hidden layers)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    all_depths = sorted(set(d for d in base_by_depth.keys()) | set(d for d in skip_by_depth.keys()))
    ax.set_xticks(all_depths)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_sine_stability_chart(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Sine Activation: Stability at Extreme Depths"
) -> None:
    """
    Show how sine maintains performance at extreme depths.
    Compare to other activations for context.
    """
    baseline = [s for s in stats if not s.skip_connections]

    # Get available activations from data
    available_activations = sorted(set(s.activation for s in baseline))

    fig, ax = plt.subplots(figsize=(10, 6))

    for act in available_activations:
        act_stats = [s for s in baseline if s.activation == act]
        if not act_stats:
            continue

        # Group by depth
        by_depth = {}
        for s in act_stats:
            if s.depth not in by_depth:
                by_depth[s.depth] = []
            by_depth[s.depth].append(s)

        depths = sorted(by_depth.keys())
        means = []
        for d in depths:
            means.append(np.mean([s.mean_accuracy for s in by_depth[d]]))

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())
        # Highlight sine/sin as the focus
        style = '-' if act in ('sine', 'sin') else '--'
        width = 3 if act in ('sine', 'sin') else 2

        ax.plot(depths, means, f'o{style}', color=color, linewidth=width,
               markersize=8, label=label)

    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Network Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_combined_phases_depth_chart(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Full Depth Range (Phases 3+4 Combined)"
) -> None:
    """
    Combined view showing depths 1-10 with Phase 3 and Phase 4 results.
    """
    baseline = [s for s in stats if not s.skip_connections]

    # Group by activation
    by_activation: Dict[str, Dict[int, List]] = {}
    for s in baseline:
        act = s.activation
        if act not in by_activation:
            by_activation[act] = {}
        if s.depth not in by_activation[act]:
            by_activation[act][s.depth] = []
        by_activation[act][s.depth].append(s.mean_accuracy)

    fig, ax = plt.subplots(figsize=(12, 7))

    for act in sorted(by_activation.keys()):
        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())

        depths = sorted(by_activation[act].keys())
        means = [np.mean(by_activation[act][d]) for d in depths]

        ax.plot(depths, means, 'o-', color=color, linewidth=2,
               markersize=7, label=label)

    # Mark Phase 3 vs Phase 4 boundary
    ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(3, 0.98, 'Phase 3', fontsize=10, ha='center', color='gray')
    ax.text(7.5, 0.98, 'Phase 4', fontsize=10, ha='center', color='gray')

    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Network Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def generate_all_phase4_plots(csv_path: str, output_dir: str) -> None:
    """
    Generate all Phase 4 visualizations from a summary CSV.
    """
    print(f"Loading results from {csv_path}...")
    stats = load_from_csv(csv_path)
    print(f"Loaded {len(stats)} configuration summaries")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("\nGenerating Phase 4 visualizations...")

    create_extended_depth_collapse(
        stats,
        str(output_path / 'phase4_depth_collapse.png')
    )

    create_skip_connection_rescue(
        stats,
        str(output_path / 'phase4_skip_rescue.png')
    )

    create_sigmoid_collapse_curve(
        stats,
        str(output_path / 'phase4_sigmoid_collapse.png')
    )

    create_sine_stability_chart(
        stats,
        str(output_path / 'phase4_sine_stability.png')
    )

    create_combined_phases_depth_chart(
        stats,
        str(output_path / 'phase4_combined_depths.png')
    )

    print(f"\nAll Phase 4 visualizations saved to: {output_path}")


def main():
    project_root = Path(__file__).parent.parent

    # Try Phase 4 summary first, then fall back to all_phases
    csv_path = project_root / 'data' / 'phase4_summary.csv'
    if not csv_path.exists():
        csv_path = project_root / 'data' / 'all_phases_summary.csv'
        if not csv_path.exists():
            print(f"Error: No summary CSV found")
            print("Run examples/aggregate_phase4.py first to generate the summary.")
            return

    output_dir = project_root / 'visualizations'
    generate_all_phase4_plots(str(csv_path), str(output_dir))


if __name__ == '__main__':
    main()
