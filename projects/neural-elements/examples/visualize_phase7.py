#!/usr/bin/env python3
"""
Generate Phase 7 visualizations: Generalization Study.

Creates charts specific to Phase 7 questions:
1. Generalization gap heatmap (activation x depth)
2. Sample efficiency curves
3. Noise robustness curves
4. Train vs test accuracy scatter
5. Dataset comparison
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


# Color schemes
ACTIVATION_COLORS = {
    'relu': '#3498db',
    'tanh': '#2ecc71',
    'gelu': '#f39c12',
    'sigmoid': '#e74c3c',
    'sine': '#9b59b6',
    'leaky_relu': '#1abc9c',
}

ACTIVATION_LABELS = {
    'relu': 'ReLU',
    'tanh': 'Tanh',
    'gelu': 'GELU',
    'sigmoid': 'Sigmoid',
    'sine': 'Sine',
    'leaky_relu': 'Leaky ReLU',
}

SAMPLE_SIZE_COLORS = {
    50: '#e74c3c',
    100: '#f39c12',
    200: '#3498db',
    500: '#2ecc71',
}


def load_phase7_csv(csv_path: str) -> List[Dict]:
    """Load Phase 7 summary CSV."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key in ('n_trials', 'depth', 'width', 'sample_size'):
                    row[key] = int(row[key]) if row[key] else 0
                elif key not in ('config_key', 'element_name', 'activation',
                                 'hidden_layers', 'dataset'):
                    try:
                        row[key] = float(row[key]) if row[key] else 0.0
                    except ValueError:
                        pass
            results.append(row)
    return results


def create_generalization_gap_heatmap(
    stats: List[Dict],
    output_path: str,
    title: str = "Generalization Gap by Activation and Depth"
) -> None:
    """
    Heatmap showing generalization gap (train - test) by activation x depth.
    Positive values (red) indicate overfitting.
    """
    # Use largest sample size for this analysis
    max_sample = max(s['sample_size'] for s in stats)
    filtered = [s for s in stats if s['sample_size'] == max_sample]

    activations = sorted(set(s['activation'] for s in filtered))
    depths = sorted(set(s['depth'] for s in filtered))

    # Create grid
    gap_grid = np.zeros((len(activations), len(depths)))
    count_grid = np.zeros((len(activations), len(depths)))

    for s in filtered:
        i = activations.index(s['activation'])
        j = depths.index(s['depth'])
        gap = s.get('generalization_gap_mean', 0) or 0
        gap_grid[i, j] += gap * 100  # Convert to percentage
        count_grid[i, j] += 1

    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        gap_grid = np.where(count_grid > 0, gap_grid / count_grid, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use diverging colormap centered at 0
    max_abs = max(abs(gap_grid.min()), abs(gap_grid.max()), 10)
    im = ax.imshow(gap_grid, cmap='RdBu_r', aspect='auto',
                   vmin=-max_abs, vmax=max_abs)

    # Labels
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f'd={d}' for d in depths])
    ax.set_yticks(range(len(activations)))
    ax.set_yticklabels([ACTIVATION_LABELS.get(a, a) for a in activations])

    # Annotations
    for i in range(len(activations)):
        for j in range(len(depths)):
            val = gap_grid[i, j]
            color = 'white' if abs(val) > max_abs * 0.5 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                    color=color, fontsize=10)

    ax.set_xlabel('Depth')
    ax.set_ylabel('Activation')
    ax.set_title(f'{title}\n(n={max_sample}, positive=overfitting)')

    plt.colorbar(im, label='Gap (train - test) %')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def create_sample_efficiency_curves(
    stats: List[Dict],
    output_path: str,
    title: str = "Sample Efficiency: Test Accuracy vs Sample Size"
) -> None:
    """
    Line chart showing how test accuracy increases with sample size.
    """
    # Group by activation and sample size
    by_act_size = defaultdict(lambda: defaultdict(list))
    for s in stats:
        act = s['activation']
        size = s['sample_size']
        test_acc = s.get('test_accuracy_mean')
        if test_acc is not None:
            by_act_size[act][size].append(test_acc * 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    sample_sizes = sorted(set(s['sample_size'] for s in stats))

    for act in sorted(by_act_size.keys()):
        sizes = []
        means = []
        stds = []
        for size in sample_sizes:
            if size in by_act_size[act]:
                vals = by_act_size[act][size]
                sizes.append(size)
                means.append(np.mean(vals))
                stds.append(np.std(vals))

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act)

        ax.plot(sizes, means, 'o-', color=color, label=label, linewidth=2, markersize=8)
        ax.fill_between(sizes,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=color, alpha=0.2)

    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([str(s) for s in sample_sizes])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def create_noise_robustness_curves(
    stats: List[Dict],
    output_path: str,
    title: str = "Noise Robustness: Test Accuracy vs Noise Level"
) -> None:
    """
    Line chart showing how test accuracy degrades with noise.
    """
    # Use largest sample size
    max_sample = max(s['sample_size'] for s in stats)
    filtered = [s for s in stats if s['sample_size'] == max_sample]

    # Group by activation
    by_act = defaultdict(lambda: defaultdict(list))
    for s in filtered:
        act = s['activation']
        for key, val in s.items():
            if key.startswith('noise_') and key.endswith('_mean'):
                noise_level = key.replace('noise_', '').replace('_mean', '')
                try:
                    noise_float = float(noise_level)
                    if val:
                        by_act[act][noise_float].append(val * 100)
                except ValueError:
                    pass

    fig, ax = plt.subplots(figsize=(10, 6))

    for act in sorted(by_act.keys()):
        noise_levels = sorted(by_act[act].keys())
        means = [np.mean(by_act[act][n]) for n in noise_levels]

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act)

        ax.plot(noise_levels, means, 'o-', color=color, label=label, linewidth=2, markersize=8)

    ax.set_xlabel('Noise Level (std)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'{title}\n(n={max_sample})')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def create_train_test_scatter(
    stats: List[Dict],
    output_path: str,
    title: str = "Train vs Test Accuracy"
) -> None:
    """
    Scatter plot of train vs test accuracy.
    Points above diagonal are overfitting.
    """
    # Use largest sample size
    max_sample = max(s['sample_size'] for s in stats)
    filtered = [s for s in stats if s['sample_size'] == max_sample]

    fig, ax = plt.subplots(figsize=(10, 8))

    for s in filtered:
        train = s.get('train_accuracy_mean')
        test = s.get('test_accuracy_mean')
        if train is not None and test is not None:
            act = s['activation']
            color = ACTIVATION_COLORS.get(act, '#888888')
            ax.scatter(train * 100, test * 100, c=color, alpha=0.6, s=50)

    # Add diagonal line (no overfitting)
    ax.plot([40, 100], [40, 100], 'k--', alpha=0.5, label='No gap')

    # Legend for activations
    for act in sorted(ACTIVATION_COLORS.keys()):
        if any(s['activation'] == act for s in filtered):
            ax.scatter([], [], c=ACTIVATION_COLORS[act],
                       label=ACTIVATION_LABELS.get(act, act))

    ax.set_xlabel('Train Accuracy (%)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'{title}\n(n={max_sample}, above diagonal = overfitting)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 105)
    ax.set_ylim(40, 105)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def create_dataset_comparison(
    stats: List[Dict],
    output_path: str,
    title: str = "Generalization by Dataset"
) -> None:
    """
    Grouped bar chart comparing generalization gap across datasets.
    """
    # Use largest sample size
    max_sample = max(s['sample_size'] for s in stats)
    filtered = [s for s in stats if s['sample_size'] == max_sample]

    # Group by dataset and activation
    by_dataset_act = defaultdict(lambda: defaultdict(list))
    for s in filtered:
        dataset = s['dataset']
        act = s['activation']
        gap = s.get('generalization_gap_mean')
        if gap is not None:
            by_dataset_act[dataset][act].append(gap * 100)

    datasets = sorted(by_dataset_act.keys())
    activations = sorted(set(s['activation'] for s in filtered))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.2
    offsets = np.linspace(-(len(activations)-1)*width/2, (len(activations)-1)*width/2, len(activations))

    for i, act in enumerate(activations):
        means = []
        for dataset in datasets:
            vals = by_dataset_act[dataset].get(act, [0])
            means.append(np.mean(vals))

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act)
        ax.bar(x + offsets[i], means, width * 0.9, color=color, label=label)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Generalization Gap (%)')
    ax.set_title(f'{title}\n(n={max_sample}, positive = overfitting)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def create_sample_size_gap_curves(
    stats: List[Dict],
    output_path: str,
    title: str = "Generalization Gap vs Sample Size"
) -> None:
    """
    Line chart showing how generalization gap changes with sample size.
    """
    # Group by activation and sample size
    by_act_size = defaultdict(lambda: defaultdict(list))
    for s in stats:
        act = s['activation']
        size = s['sample_size']
        gap = s.get('generalization_gap_mean')
        if gap is not None:
            by_act_size[act][size].append(gap * 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    sample_sizes = sorted(set(s['sample_size'] for s in stats))

    for act in sorted(by_act_size.keys()):
        sizes = []
        means = []
        for size in sample_sizes:
            if size in by_act_size[act]:
                vals = by_act_size[act][size]
                sizes.append(size)
                means.append(np.mean(vals))

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act)

        ax.plot(sizes, means, 'o-', color=color, label=label, linewidth=2, markersize=8)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No gap')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Generalization Gap (%)')
    ax.set_title(f'{title}\n(positive = overfitting)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([str(s) for s in sample_sizes])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    viz_path = project_root / 'visualizations'
    viz_path.mkdir(exist_ok=True)

    # Load summary CSV
    csv_path = data_path / 'phase7_summary.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        print("Run aggregate_phase7.py first")
        return

    print("=" * 60)
    print("PHASE 7 VISUALIZATIONS: Generalization Study")
    print("=" * 60)

    stats = load_phase7_csv(str(csv_path))
    print(f"\nLoaded {len(stats)} configuration summaries")

    if not stats:
        print("No data to visualize")
        return

    # Generate all charts
    print("\nGenerating visualizations...")

    # 1. Generalization gap heatmap
    create_generalization_gap_heatmap(
        stats,
        str(viz_path / 'phase7_generalization_gap_heatmap.png')
    )

    # 2. Sample efficiency curves
    create_sample_efficiency_curves(
        stats,
        str(viz_path / 'phase7_sample_efficiency_curves.png')
    )

    # 3. Noise robustness curves
    create_noise_robustness_curves(
        stats,
        str(viz_path / 'phase7_noise_robustness_curves.png')
    )

    # 4. Train vs test scatter
    create_train_test_scatter(
        stats,
        str(viz_path / 'phase7_train_test_scatter.png')
    )

    # 5. Dataset comparison
    create_dataset_comparison(
        stats,
        str(viz_path / 'phase7_dataset_comparison.png')
    )

    # 6. Gap vs sample size
    create_sample_size_gap_curves(
        stats,
        str(viz_path / 'phase7_gap_vs_sample_size.png')
    )

    print("\n" + "=" * 60)
    print("Phase 7 visualizations complete!")
    print(f"Charts saved to: {viz_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
