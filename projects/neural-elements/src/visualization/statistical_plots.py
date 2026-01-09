"""
Statistical visualization module for Phase 3+ results.

Creates publication-quality charts with:
- Confidence intervals and error bars
- Statistical significance annotations
- Proper uncertainty visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path

# Import from analysis module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.analysis import ConfigurationStats, load_from_csv


# Color schemes
ACTIVATION_COLORS = {
    'relu': '#3498db',      # Blue
    'rel': '#3498db',
    'tanh': '#2ecc71',      # Green
    'tan': '#2ecc71',
    'gelu': '#f39c12',      # Orange
    'gel': '#f39c12',
    'sigmoid': '#e74c3c',   # Red
    'sig': '#e74c3c',
    'sine': '#9b59b6',      # Purple
    'sin': '#9b59b6',
    'leaky_relu': '#1abc9c', # Teal
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


def create_depth_effect_with_ci(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Depth Effect by Activation (with 95% CI)"
) -> None:
    """
    Create depth effect chart with shaded confidence intervals.

    Shows how each activation responds to increasing depth,
    with uncertainty bands around each line.
    """
    # Group by activation
    by_activation: Dict[str, Dict[int, ConfigurationStats]] = {}

    for stat in stats:
        act = stat.activation
        if act not in by_activation:
            by_activation[act] = {}
        # Average across datasets for this activation+depth
        if stat.depth not in by_activation[act]:
            by_activation[act][stat.depth] = []
        by_activation[act][stat.depth].append(stat)

    fig, ax = plt.subplots(figsize=(10, 6))

    depths = sorted(set(s.depth for s in stats))

    for act in ['sigmoid', 'sig', 'relu', 'rel', 'tanh', 'tan', 'gelu', 'gel', 'sine', 'sin', 'leaky_relu', 'lea']:
        if act not in by_activation:
            continue

        # Normalize activation name
        act_key = act[:3] if len(act) > 3 else act
        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())

        means = []
        ci_lowers = []
        ci_uppers = []
        valid_depths = []

        for d in depths:
            if d in by_activation[act]:
                stat_list = by_activation[act][d]
                # Combine stats across datasets
                all_means = [s.mean_accuracy for s in stat_list]
                mean_acc = np.mean(all_means)
                # Use average CI width
                ci_lower = mean_acc - np.mean([s.mean_accuracy - s.ci_lower for s in stat_list])
                ci_upper = mean_acc + np.mean([s.ci_upper - s.mean_accuracy for s in stat_list])

                means.append(mean_acc)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
                valid_depths.append(d)

        if not means:
            continue

        # Plot shaded CI region
        ax.fill_between(valid_depths, ci_lowers, ci_uppers,
                       color=color, alpha=0.2)
        # Plot mean line
        ax.plot(valid_depths, means, 'o-', color=color, linewidth=2.5,
               markersize=8, label=label)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')

    ax.set_xlabel('Network Depth (layers)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(depths)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_activation_ranking_with_ci(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Activation Ranking (with 95% CI)"
) -> None:
    """
    Create horizontal bar chart of activation ranking with error bars.
    """
    # Aggregate by activation
    by_activation: Dict[str, List[float]] = {}
    ci_data: Dict[str, List[tuple]] = {}

    for stat in stats:
        act = stat.activation
        if act not in by_activation:
            by_activation[act] = []
            ci_data[act] = []
        by_activation[act].append(stat.mean_accuracy)
        ci_data[act].append((stat.ci_lower, stat.ci_upper))

    # Calculate overall mean and CI for each activation
    activation_stats = []
    for act, accs in by_activation.items():
        mean_acc = np.mean(accs)
        # Pooled CI (approximate)
        ci_widths = [(u - l) / 2 for l, u in ci_data[act]]
        avg_ci_width = np.mean(ci_widths)
        activation_stats.append({
            'activation': act,
            'mean': mean_acc,
            'ci_lower': mean_acc - avg_ci_width,
            'ci_upper': mean_acc + avg_ci_width,
            'ci_width': avg_ci_width,
        })

    # Sort by mean accuracy
    activation_stats.sort(key=lambda x: x['mean'], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(activation_stats))
    means = [s['mean'] for s in activation_stats]
    errors = [s['ci_width'] for s in activation_stats]
    colors = [ACTIVATION_COLORS.get(s['activation'], '#888888') for s in activation_stats]
    labels = [ACTIVATION_LABELS.get(s['activation'], s['activation'].upper()) for s in activation_stats]

    bars = ax.barh(y_pos, means, xerr=errors, color=colors,
                   capsize=5, error_kw={'linewidth': 2})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, stat) in enumerate(zip(bars, activation_stats)):
        ax.text(stat['mean'] + stat['ci_width'] + 0.01, i,
               f"{stat['mean']*100:.1f}%", va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_dataset_comparison_with_ci(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Performance by Dataset (with 95% CI)"
) -> None:
    """
    Create 2x2 subplot comparing activations on each dataset with error bars.
    """
    datasets = ['xor', 'moons', 'circles', 'spirals']
    dataset_titles = {
        'xor': 'XOR (Classic Test)',
        'moons': 'Moons (Curved Boundary)',
        'circles': 'Circles (Radial)',
        'spirals': 'Spirals (Hard)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Get activation order (sorted by overall performance)
    activation_order = ['leaky_relu', 'relu', 'sine', 'tanh', 'gelu', 'sigmoid']
    # Normalize to 3-letter codes
    activation_order = [a[:3] if len(a) <= 3 else a for a in activation_order]

    for ax, dataset in zip(axes.flat, datasets):
        # Get stats for this dataset
        dataset_stats = [s for s in stats if s.dataset == dataset]

        # Group by activation, aggregate across depths
        by_activation: Dict[str, Dict] = {}
        for stat in dataset_stats:
            act = stat.activation
            if act not in by_activation:
                by_activation[act] = {'means': [], 'ci_widths': []}
            by_activation[act]['means'].append(stat.mean_accuracy)
            by_activation[act]['ci_widths'].append((stat.ci_upper - stat.ci_lower) / 2)

        # Calculate aggregated stats
        activations = []
        means = []
        errors = []
        colors = []

        for act in activation_order:
            if act in by_activation:
                activations.append(ACTIVATION_LABELS.get(act, act.upper()))
                means.append(np.mean(by_activation[act]['means']))
                errors.append(np.mean(by_activation[act]['ci_widths']))
                colors.append(ACTIVATION_COLORS.get(act, '#888888'))

        x_pos = np.arange(len(activations))
        bars = ax.bar(x_pos, means, yerr=errors, color=colors,
                     capsize=4, error_kw={'linewidth': 1.5})

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel('Accuracy')
        ax.set_title(dataset_titles[dataset], fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(activations, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight best
        if means:
            max_idx = np.argmax(means)
            bars[max_idx].set_edgecolor('black')
            bars[max_idx].set_linewidth(2)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_variance_reliability_plot(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Accuracy vs Reliability (Lower std = more reliable)"
) -> None:
    """
    Scatter plot of mean accuracy vs standard deviation.

    Helps identify which configurations are both accurate AND consistent.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for stat in stats:
        color = ACTIVATION_COLORS.get(stat.activation, '#888888')
        # Size based on depth (deeper = larger)
        size = 50 + stat.depth * 30

        ax.scatter(stat.std_accuracy, stat.mean_accuracy,
                  c=color, s=size, alpha=0.6, edgecolors='white', linewidth=0.5)

    # Add legend for activations
    handles = []
    for act in ['relu', 'tanh', 'gelu', 'sigmoid', 'sine', 'leaky_relu']:
        if act in [s.activation for s in stats]:
            color = ACTIVATION_COLORS.get(act, '#888888')
            label = ACTIVATION_LABELS.get(act, act.upper())
            handles.append(plt.scatter([], [], c=color, s=100, label=label))

    ax.legend(handles=handles, title='Activation', loc='lower left')

    ax.set_xlabel('Standard Deviation (lower = more reliable)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.01, max(s.std_accuracy for s in stats) * 1.1)
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)

    # Add "Pareto frontier" annotation region
    ax.axhspan(0.9, 1.02, xmin=0, xmax=0.3, alpha=0.1, color='green')
    ax.text(0.02, 0.95, 'Best: High accuracy,\nlow variance', fontsize=10,
           color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_box_plots_by_activation(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Accuracy Distribution by Activation"
) -> None:
    """
    Box plots showing the full distribution of accuracies for each activation.
    """
    # Group by activation
    by_activation: Dict[str, List[float]] = {}

    for stat in stats:
        act = stat.activation
        if act not in by_activation:
            by_activation[act] = []
        by_activation[act].append(stat.mean_accuracy)

    # Sort by median
    sorted_activations = sorted(by_activation.keys(),
                                key=lambda a: np.median(by_activation[a]),
                                reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    data = [by_activation[act] for act in sorted_activations]
    labels = [ACTIVATION_LABELS.get(act, act.upper()) for act in sorted_activations]
    colors = [ACTIVATION_COLORS.get(act, '#888888') for act in sorted_activations]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def generate_all_phase3_plots(csv_path: str, output_dir: str) -> None:
    """
    Generate all Phase 3 statistical plots from a summary CSV.
    """
    print(f"Loading results from {csv_path}...")
    stats = load_from_csv(csv_path)
    print(f"Loaded {len(stats)} configuration summaries")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("\nGenerating Phase 3 visualizations with confidence intervals...")

    create_depth_effect_with_ci(
        stats,
        str(output_path / 'phase3_depth_effect_ci.png')
    )

    create_activation_ranking_with_ci(
        stats,
        str(output_path / 'phase3_activation_ranking_ci.png')
    )

    create_dataset_comparison_with_ci(
        stats,
        str(output_path / 'phase3_dataset_comparison_ci.png')
    )

    create_variance_reliability_plot(
        stats,
        str(output_path / 'phase3_variance_reliability.png')
    )

    create_box_plots_by_activation(
        stats,
        str(output_path / 'phase3_activation_boxplots.png')
    )

    print(f"\nAll Phase 3 visualizations saved to: {output_path}")
