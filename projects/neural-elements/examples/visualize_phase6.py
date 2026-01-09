#!/usr/bin/env python3
"""
Generate Phase 6 visualizations: Learning Dynamics.

Creates charts specific to Phase 6 questions:
1. Convergence speed comparison by activation
2. Gradient flow over training
3. Training stability heatmap
4. Skip connection dynamics comparison
5. Bottleneck gradient flow
6. Vanishing gradient detection scatter
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


# Color schemes
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


def load_phase6_csv(csv_path: str) -> List[Dict]:
    """Load Phase 6 summary CSV."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key in ('n_trials', 'depth', 'width', 'min_width'):
                    row[key] = int(row[key]) if row[key] else 0
                elif key == 'skip_connections':
                    row[key] = row[key].lower() == 'true'
                elif key not in ('config_key', 'element_name', 'activation',
                                 'hidden_layers', 'architecture_type', 'dataset'):
                    try:
                        row[key] = float(row[key]) if row[key] else 0.0
                    except ValueError:
                        pass
            results.append(row)
    return results


def create_convergence_comparison(
    stats: List[Dict],
    output_path: str,
    title: str = "Convergence Speed by Activation and Depth"
) -> None:
    """
    Bar chart showing epochs-to-90% by activation, grouped by depth.
    """
    # Filter to baseline only
    baseline = [s for s in stats if not s.get('skip_connections', False)
                and s.get('architecture_type') == 'uniform']

    # Group by activation and depth
    by_act_depth = defaultdict(lambda: defaultdict(list))
    for s in baseline:
        act = s['activation']
        depth = s['depth']
        conv = s.get('convergence_epoch_mean', 0)
        if conv and conv > 0:
            by_act_depth[act][depth].append(conv)

    if not by_act_depth:
        print(f"No convergence data found - skipping {output_path}")
        return

    activations = sorted(by_act_depth.keys())
    depths = sorted(set(d for act in by_act_depth.values() for d in act.keys()))

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(depths))
    width = 0.8 / len(activations)

    for i, act in enumerate(activations):
        means = [np.mean(by_act_depth[act].get(d, [0])) for d in depths]
        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())
        bars = ax.bar(x + i * width - 0.4 + width/2, means, width,
                     label=label, color=color, alpha=0.8)

    ax.set_xlabel('Network Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Epochs to 90% Final Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'd={d}' for d in depths])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_stability_heatmap(
    stats: List[Dict],
    output_path: str,
    title: str = "Training Stability (Loss Variance in Final 20%)"
) -> None:
    """
    Heatmap showing training stability by activation × depth.
    Lower = more stable.
    """
    # Filter to baseline only
    baseline = [s for s in stats if not s.get('skip_connections', False)
                and s.get('architecture_type') == 'uniform']

    # Build matrix
    activations = sorted(set(s['activation'] for s in baseline))
    depths = sorted(set(s['depth'] for s in baseline))

    matrix = np.zeros((len(activations), len(depths)))
    for s in baseline:
        i = activations.index(s['activation'])
        j = depths.index(s['depth'])
        stability = s.get('loss_stability_mean', 0)
        matrix[i, j] = stability

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for stability values (which can vary widely)
    matrix_log = np.log10(matrix + 1e-6)

    im = ax.imshow(matrix_log, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f'd={d}' for d in depths])
    ax.set_yticks(range(len(activations)))
    ax.set_yticklabels([ACTIVATION_LABELS.get(a, a.upper()) for a in activations])

    # Add values
    for i in range(len(activations)):
        for j in range(len(depths)):
            val = matrix[i, j]
            text_color = 'white' if matrix_log[i, j] > -2 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                   fontsize=9, color=text_color)

    ax.set_xlabel('Network Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Activation', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log10(Loss Stability)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_skip_dynamics_comparison(
    stats: List[Dict],
    output_path: str,
    title: str = "Skip Connection Effect on Training Dynamics"
) -> None:
    """
    2x2 grid comparing baseline vs skip for each activation.
    Shows both accuracy and gradient health.
    """
    activations = ['relu', 'sigmoid', 'sine', 'tanh']

    # Try to find both canonical names and 3-letter codes
    def find_stats(stats_list, act, skip, depths):
        codes = [act, act[:3]]
        return [s for s in stats_list
                if s['activation'] in codes
                and s.get('skip_connections', False) == skip
                and s['depth'] in depths]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    depths = [5, 8]

    for ax, act in zip(axes.flat, activations):
        baseline = find_stats(stats, act, False, depths)
        with_skip = find_stats(stats, act, True, depths)

        if not baseline or not with_skip:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{ACTIVATION_LABELS.get(act, act.upper())}')
            continue

        # Group by depth
        base_by_depth = defaultdict(list)
        skip_by_depth = defaultdict(list)

        for s in baseline:
            base_by_depth[s['depth']].append(s)
        for s in with_skip:
            skip_by_depth[s['depth']].append(s)

        x = np.arange(len(depths))
        width = 0.35

        # Accuracy bars
        base_acc = [np.mean([s['mean_accuracy'] for s in base_by_depth[d]]) for d in depths]
        skip_acc = [np.mean([s['mean_accuracy'] for s in skip_by_depth[d]]) for d in depths]

        color = ACTIVATION_COLORS.get(act, '#888888')

        bars1 = ax.bar(x - width/2, base_acc, width, label='Baseline',
                      color=color, alpha=0.7)
        bars2 = ax.bar(x + width/2, skip_acc, width, label='With Skip',
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
            if base_acc[i] > 0 and skip_acc[i] > 0:
                delta = (skip_acc[i] - base_acc[i]) * 100
                if abs(delta) > 1:
                    symbol = '+' if delta > 0 else ''
                    color_txt = 'green' if delta > 0 else 'red'
                    ax.text(i, max(base_acc[i], skip_acc[i]) + 0.02,
                           f'{symbol}{delta:.0f}%', ha='center', fontsize=9,
                           fontweight='bold', color=color_txt)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_bottleneck_dynamics(
    stats: List[Dict],
    output_path: str,
    title: str = "Bottleneck Architecture Dynamics"
) -> None:
    """
    Compare dynamics across bottleneck architectures.
    """
    # Find bottleneck configurations
    bottleneck = [s for s in stats if s.get('architecture_type') == 'bottleneck']
    uniform = [s for s in stats if s.get('architecture_type') == 'uniform' and s['depth'] == 3]

    if not bottleneck:
        print(f"No bottleneck data found - skipping {output_path}")
        return

    # Group by hidden_layers
    by_arch = defaultdict(list)
    for s in bottleneck:
        by_arch[s['hidden_layers']].append(s)
    for s in uniform:
        by_arch['[8, 8, 8]'].append(s)

    architectures = sorted(by_arch.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1. Accuracy comparison
    ax = axes[0]
    arch_means = []
    for arch in architectures:
        if by_arch[arch]:
            mean_acc = np.mean([s['mean_accuracy'] for s in by_arch[arch]])
            arch_means.append((arch, mean_acc))

    if arch_means:
        labels, values = zip(*arch_means)
        colors = ['#e74c3c' if '2' in l else '#3498db' for l in labels]
        ax.barh(range(len(labels)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Mean Accuracy')
        ax.set_title('Accuracy by Architecture')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(0.4, 1.0)

    # 2. Convergence speed
    ax = axes[1]
    arch_conv = []
    for arch in architectures:
        if by_arch[arch]:
            convs = [s.get('convergence_epoch_mean', 0) for s in by_arch[arch]]
            if convs:
                arch_conv.append((arch, np.mean(convs)))

    if arch_conv:
        labels, values = zip(*arch_conv)
        colors = ['#e74c3c' if '2' in l else '#3498db' for l in labels]
        ax.barh(range(len(labels)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Epochs to 90%')
        ax.set_title('Convergence Speed')

    # 3. Gradient ratio (late/early)
    ax = axes[2]
    arch_grad = []
    for arch in architectures:
        if by_arch[arch]:
            ratios = [s.get('grad_gradient_ratio_mean', 1) for s in by_arch[arch]]
            if ratios:
                arch_grad.append((arch, np.mean(ratios)))

    if arch_grad:
        labels, values = zip(*arch_grad)
        colors = ['#e74c3c' if v < 0.5 else '#2ecc71' for v in values]
        ax.barh(range(len(labels)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Gradient Ratio (late/early)')
        ax.set_title('Gradient Health')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_vanishing_gradient_scatter(
    stats: List[Dict],
    output_path: str,
    title: str = "Vanishing Gradients vs Accuracy"
) -> None:
    """
    Scatter plot: final gradient norm vs final accuracy.
    Points colored by activation, sized by depth.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for s in stats:
        grad_final = s.get('grad_mean_grad_final_mean', 0)
        accuracy = s.get('mean_accuracy', 0)

        if grad_final > 0 and accuracy > 0:
            color = ACTIVATION_COLORS.get(s['activation'], '#888888')
            size = s['depth'] * 20  # Size by depth

            ax.scatter(grad_final, accuracy, c=color, s=size, alpha=0.6)

    # Add legend for activations
    for act in ['relu', 'sigmoid', 'sine', 'tanh']:
        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())
        ax.scatter([], [], c=color, s=100, label=label, alpha=0.8)

    ax.set_xscale('log')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    ax.axvline(x=1e-6, color='red', linestyle=':', alpha=0.5, label='Vanishing threshold')

    ax.set_xlabel('Final Mean Gradient (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_activation_dynamics_summary(
    stats: List[Dict],
    output_path: str,
    title: str = "Activation Function Dynamics Summary"
) -> None:
    """
    Combined view showing key dynamics metrics for each activation.
    """
    # Filter to baseline uniform
    baseline = [s for s in stats if not s.get('skip_connections', False)
                and s.get('architecture_type') == 'uniform']

    # Aggregate by activation
    by_activation = defaultdict(list)
    for s in baseline:
        by_activation[s['activation']].append(s)

    activations = sorted(by_activation.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Accuracy vs Depth
    ax = axes[0, 0]
    for act in activations:
        by_depth = defaultdict(list)
        for s in by_activation[act]:
            by_depth[s['depth']].append(s['mean_accuracy'])

        depths = sorted(by_depth.keys())
        means = [np.mean(by_depth[d]) for d in depths]

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())
        ax.plot(depths, means, 'o-', color=color, label=label, linewidth=2, markersize=6)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Depth')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Convergence vs Depth
    ax = axes[0, 1]
    for act in activations:
        by_depth = defaultdict(list)
        for s in by_activation[act]:
            conv = s.get('convergence_epoch_mean', 0)
            if conv and conv > 0:
                by_depth[s['depth']].append(conv)

        if by_depth:
            depths = sorted(by_depth.keys())
            means = [np.mean(by_depth[d]) for d in depths]

            color = ACTIVATION_COLORS.get(act, '#888888')
            label = ACTIVATION_LABELS.get(act, act.upper())
            ax.plot(depths, means, 'o-', color=color, label=label, linewidth=2, markersize=6)

    ax.set_xlabel('Depth')
    ax.set_ylabel('Epochs to 90%')
    ax.set_title('Convergence Speed vs Depth')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Gradient ratio vs Depth
    ax = axes[1, 0]
    for act in activations:
        by_depth = defaultdict(list)
        for s in by_activation[act]:
            ratio = s.get('grad_gradient_ratio_mean', 0)
            if ratio and ratio > 0:
                by_depth[s['depth']].append(ratio)

        if by_depth:
            depths = sorted(by_depth.keys())
            means = [np.mean(by_depth[d]) for d in depths]

            color = ACTIVATION_COLORS.get(act, '#888888')
            label = ACTIVATION_LABELS.get(act, act.upper())
            ax.plot(depths, means, 'o-', color=color, label=label, linewidth=2, markersize=6)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Gradient Ratio (late/early)')
    ax.set_title('Gradient Flow Health')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 4. Vanishing rate vs Depth
    ax = axes[1, 1]
    for act in activations:
        by_depth = defaultdict(list)
        for s in by_activation[act]:
            rate = s.get('vanishing_gradient_rate', 0)
            by_depth[s['depth']].append(rate)

        if by_depth:
            depths = sorted(by_depth.keys())
            means = [np.mean(by_depth[d]) * 100 for d in depths]

            color = ACTIVATION_COLORS.get(act, '#888888')
            label = ACTIVATION_LABELS.get(act, act.upper())
            ax.plot(depths, means, 'o-', color=color, label=label, linewidth=2, markersize=6)

    ax.set_xlabel('Depth')
    ax.set_ylabel('Vanishing Gradient Rate (%)')
    ax.set_title('Vanishing Gradient Occurrence')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def generate_all_phase6_plots(csv_path: str, output_dir: str) -> None:
    """
    Generate all Phase 6 visualizations from a summary CSV.
    """
    print(f"Loading results from {csv_path}...")
    stats = load_phase6_csv(csv_path)
    print(f"Loaded {len(stats)} configuration summaries")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("\nGenerating Phase 6 visualizations...")

    create_convergence_comparison(
        stats,
        str(output_path / 'phase6_convergence_comparison.png')
    )

    create_stability_heatmap(
        stats,
        str(output_path / 'phase6_stability_heatmap.png')
    )

    create_skip_dynamics_comparison(
        stats,
        str(output_path / 'phase6_skip_dynamics.png')
    )

    create_bottleneck_dynamics(
        stats,
        str(output_path / 'phase6_bottleneck_dynamics.png')
    )

    create_vanishing_gradient_scatter(
        stats,
        str(output_path / 'phase6_vanishing_gradient_detection.png')
    )

    create_activation_dynamics_summary(
        stats,
        str(output_path / 'phase6_activation_dynamics_summary.png')
    )

    print(f"\nAll Phase 6 visualizations saved to: {output_path}")


def main():
    project_root = Path(__file__).parent.parent

    csv_path = project_root / 'data' / 'phase6_summary.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        print("Run examples/aggregate_phase6.py first to generate the summary.")
        return

    output_dir = project_root / 'visualizations'
    generate_all_phase6_plots(str(csv_path), str(output_dir))


if __name__ == '__main__':
    main()
