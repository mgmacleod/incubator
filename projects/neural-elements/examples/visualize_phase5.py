#!/usr/bin/env python3
"""
Generate Phase 5 visualizations: Architecture Space Exploration.

Creates charts showing:
1. Bottleneck survival - which bottleneck widths can still learn?
2. Pyramid comparison - expanding vs contracting vs diamond
3. Parameter efficiency - accuracy per parameter count
4. Architecture heatmap - pattern × activation performance grid
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import load_from_csv, ConfigurationStats, Phase5Config
from typing import List, Dict, Optional


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

# Architecture pattern colors
PATTERN_COLORS = {
    'uniform': '#3498db',
    'bottleneck': '#e74c3c',
    'expanding': '#2ecc71',
    'contracting': '#f39c12',
    'diamond': '#9b59b6',
    'wide_shallow': '#1abc9c',
    'narrow_deep': '#e67e22',
}


def get_architecture_pattern(hidden_layers: List[int]) -> str:
    """Classify hidden_layers into a pattern."""
    if not hidden_layers:
        return 'linear'

    if all(w == hidden_layers[0] for w in hidden_layers):
        return 'uniform'

    # Bottleneck: min in middle
    if len(hidden_layers) >= 3:
        middle = len(hidden_layers) // 2
        if hidden_layers[middle] < hidden_layers[0] and hidden_layers[middle] < hidden_layers[-1]:
            return 'bottleneck'

    if hidden_layers == sorted(hidden_layers):
        return 'expanding'
    if hidden_layers == sorted(hidden_layers, reverse=True):
        return 'contracting'

    # Diamond
    peak_idx = hidden_layers.index(max(hidden_layers))
    if 0 < peak_idx < len(hidden_layers) - 1:
        return 'diamond'

    return 'mixed'


def match_stat_to_architecture(stat: ConfigurationStats, config: Phase5Config) -> Optional[str]:
    """Try to match a stat to a known architecture from config."""
    for name, layers in config.architectures.items():
        if stat.depth == len(layers) and stat.width == max(layers):
            return name
    return None


def create_bottleneck_survival_chart(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Bottleneck Architecture Survival"
) -> None:
    """Show which bottleneck widths can still learn."""
    # Known bottleneck architectures
    bottlenecks = {
        '[32, 8, 32]': {'depth': 3, 'width': 32, 'min_width': 8},
        '[16, 4, 16]': {'depth': 3, 'width': 16, 'min_width': 4},
        '[8, 2, 8]': {'depth': 3, 'width': 8, 'min_width': 2},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = list(bottlenecks.keys())
    x_pos = np.arange(len(x_labels))

    activations = sorted(set(s.activation for s in stats))

    bar_width = 0.15
    offset = (len(activations) - 1) * bar_width / 2

    for i, act in enumerate(activations):
        means = []
        errors = []

        for arch_name, arch_info in bottlenecks.items():
            # Find matching stats
            matches = [s for s in stats
                       if s.activation == act
                       and s.depth == arch_info['depth']
                       and s.width == arch_info['width']]

            if matches:
                avg = np.mean([s.mean_accuracy for s in matches])
                std = np.mean([s.std_accuracy for s in matches])
                means.append(avg)
                errors.append(std)
            else:
                means.append(0)
                errors.append(0)

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())

        ax.bar(x_pos + i * bar_width - offset, means, bar_width,
               yerr=errors, label=label, color=color, capsize=3, alpha=0.8)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')

    ax.set_xlabel('Bottleneck Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation about width=2 bottleneck
    ax.annotate('Width-2 bottleneck\n(Phase 3 showed this is devastating)',
                xy=(2, 0.55), xytext=(1.5, 0.65),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_pyramid_comparison_chart(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Pyramid Architecture Comparison"
) -> None:
    """Compare expanding vs contracting vs diamond pyramids."""
    pyramids = {
        'Expanding\n[4, 8, 16]': {'depth': 3, 'width': 16},
        'Contracting\n[16, 8, 4]': {'depth': 3, 'width': 16},
        'Diamond\n[4, 8, 16, 8, 4]': {'depth': 5, 'width': 16},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = list(pyramids.keys())
    x_pos = np.arange(len(x_labels))

    activations = sorted(set(s.activation for s in stats))

    bar_width = 0.15
    offset = (len(activations) - 1) * bar_width / 2

    for i, act in enumerate(activations):
        means = []
        errors = []

        for arch_name, arch_info in pyramids.items():
            matches = [s for s in stats
                       if s.activation == act
                       and s.depth == arch_info['depth']
                       and s.width == arch_info['width']]

            if matches:
                avg = np.mean([s.mean_accuracy for s in matches])
                std = np.mean([s.std_accuracy for s in matches])
                means.append(avg)
                errors.append(std)
            else:
                means.append(0)
                errors.append(0)

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())

        ax.bar(x_pos + i * bar_width - offset, means, bar_width,
               yerr=errors, label=label, color=color, capsize=3, alpha=0.8)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Pyramid Pattern', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_parameter_efficiency_chart(
    stats: List[ConfigurationStats],
    output_path: str,
    title: str = "Parameter Efficiency: Shape vs Size"
) -> None:
    """Compare architectures with similar parameter counts."""
    # Architectures with roughly similar param counts
    param_matched = {
        'Wide-Shallow\n[32]': {'depth': 1, 'width': 32, 'params': 97},
        'Medium\n[12, 12]': {'depth': 2, 'width': 12, 'params': 193},
        'Narrow-Deep\n[8, 8, 8]': {'depth': 3, 'width': 8, 'params': 169},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = list(param_matched.keys())
    x_pos = np.arange(len(x_labels))

    activations = sorted(set(s.activation for s in stats))

    bar_width = 0.15
    offset = (len(activations) - 1) * bar_width / 2

    for i, act in enumerate(activations):
        means = []
        errors = []

        for arch_name, arch_info in param_matched.items():
            matches = [s for s in stats
                       if s.activation == act
                       and s.depth == arch_info['depth']
                       and s.width == arch_info['width']]

            if matches:
                avg = np.mean([s.mean_accuracy for s in matches])
                std = np.mean([s.std_accuracy for s in matches])
                means.append(avg)
                errors.append(std)
            else:
                means.append(0)
                errors.append(0)

        color = ACTIVATION_COLORS.get(act, '#888888')
        label = ACTIVATION_LABELS.get(act, act.upper())

        ax.bar(x_pos + i * bar_width - offset, means, bar_width,
               yerr=errors, label=label, color=color, capsize=3, alpha=0.8)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Architecture (similar param counts)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add param count annotations
    for j, (name, info) in enumerate(param_matched.items()):
        ax.annotate(f"~{info['params']} params",
                    xy=(j, 0.42), ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_architecture_heatmap(
    stats: List[ConfigurationStats],
    config: Phase5Config,
    output_path: str,
    title: str = "Architecture × Activation Performance"
) -> None:
    """Create a heatmap of architecture vs activation performance."""
    arch_names = list(config.architectures.keys())
    activations = sorted(set(s.activation for s in stats))

    # Create matrix
    matrix = np.zeros((len(arch_names), len(activations)))

    for i, arch_name in enumerate(arch_names):
        arch_layers = config.architectures[arch_name]
        arch_depth = len(arch_layers)
        arch_width = max(arch_layers)

        for j, act in enumerate(activations):
            matches = [s for s in stats
                       if s.activation == act
                       and s.depth == arch_depth
                       and s.width == arch_width]
            if matches:
                matrix[i, j] = np.mean([s.mean_accuracy for s in matches])
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

    # Labels
    ax.set_xticks(np.arange(len(activations)))
    ax.set_yticks(np.arange(len(arch_names)))
    ax.set_xticklabels([ACTIVATION_LABELS.get(a, a.upper()) for a in activations], fontsize=10)
    ax.set_yticklabels(arch_names, fontsize=9)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Average Accuracy', fontsize=11)

    # Add value annotations
    for i in range(len(arch_names)):
        for j in range(len(activations)):
            if not np.isnan(matrix[i, j]):
                text = f'{matrix[i, j]*100:.0f}%'
                color = 'white' if matrix[i, j] < 0.7 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

    ax.set_xlabel('Activation Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Architecture', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_dataset_comparison(
    stats: List[ConfigurationStats],
    config: Phase5Config,
    output_path: str,
    title: str = "Performance by Dataset"
) -> None:
    """Show performance breakdown by dataset for key architectures."""
    # Select representative architectures
    key_archs = ['bottleneck_severe', 'pyramid_expanding', 'wide_shallow', 'narrow_deep']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    datasets = ['xor', 'moons', 'circles', 'spirals']

    for ax, arch_name in zip(axes.flat, key_archs):
        if arch_name not in config.architectures:
            continue

        arch_layers = config.architectures[arch_name]
        arch_depth = len(arch_layers)
        arch_width = max(arch_layers)

        # Get stats for this architecture
        arch_stats = [s for s in stats if s.depth == arch_depth and s.width == arch_width]

        activations = sorted(set(s.activation for s in arch_stats))
        x_pos = np.arange(len(datasets))
        bar_width = 0.2
        offset = (len(activations) - 1) * bar_width / 2

        for i, act in enumerate(activations):
            means = []
            for dataset in datasets:
                matches = [s for s in arch_stats if s.activation == act and s.dataset == dataset]
                if matches:
                    means.append(np.mean([s.mean_accuracy for s in matches]))
                else:
                    means.append(0)

            color = ACTIVATION_COLORS.get(act, '#888888')
            label = ACTIVATION_LABELS.get(act, act.upper())
            ax.bar(x_pos + i * bar_width - offset, means, bar_width, label=label, color=color, alpha=0.8)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0.4, 1.05)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_title(f'{arch_name}\n{arch_layers}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / 'data' / 'phase5_summary.csv'
    output_dir = project_root / 'visualizations'

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        print("Run examples/aggregate_phase5.py first to generate the summary.")
        return

    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from {csv_path}...")
    stats = load_from_csv(str(csv_path))
    print(f"Loaded {len(stats)} configuration summaries")

    config = Phase5Config()

    print("\nGenerating Phase 5 visualizations...")

    create_bottleneck_survival_chart(
        stats,
        str(output_dir / 'phase5_bottleneck_survival.png')
    )

    create_pyramid_comparison_chart(
        stats,
        str(output_dir / 'phase5_pyramid_comparison.png')
    )

    create_parameter_efficiency_chart(
        stats,
        str(output_dir / 'phase5_parameter_efficiency.png')
    )

    create_architecture_heatmap(
        stats,
        config,
        str(output_dir / 'phase5_architecture_heatmap.png')
    )

    create_dataset_comparison(
        stats,
        config,
        str(output_dir / 'phase5_dataset_comparison.png')
    )

    print(f"\nAll Phase 5 visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
