#!/usr/bin/env python3
"""
Generate Phase 8 visualizations: Combination & Transfer.

Creates charts specific to Phase 8 questions:
1. Stacking heatmap (bottom x top activation/depth)
2. Transfer matrix (source -> target)
3. Ensemble scaling (size vs improvement)
4. Combined composition summary
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
    'sigmoid': '#e74c3c',
    'sine': '#9b59b6',
}

ACTIVATION_LABELS = {
    'relu': 'ReLU',
    'tanh': 'Tanh',
    'sigmoid': 'Sigmoid',
    'sine': 'Sine',
}

ENSEMBLE_COLORS = {
    'voting': '#3498db',
    'averaging': '#2ecc71',
    'weighted': '#f39c12',
}


def load_csv(csv_path: str) -> List[Dict]:
    """Load CSV file."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key not in ('experiment_type', 'activation', 'dataset',
                               'source_dataset', 'target_dataset', 'freeze_mode',
                               'ensemble_type'):
                    try:
                        row[key] = float(row[key]) if row[key] else None
                    except ValueError:
                        pass
            results.append(row)
    return results


def create_stacking_heatmap(
    stacking: List[Dict],
    output_path: str,
    title: str = "Stacking: Improvement by Bottom/Top Depth"
) -> None:
    """Heatmap showing stacking improvement by bottom x top depth."""
    if not stacking:
        print(f"   Skipping stacking heatmap (no data)")
        return

    # Group by activation
    by_activation = defaultdict(list)
    for s in stacking:
        by_activation[s['activation']].append(s)

    # Create subplot for each activation
    n_activations = len(by_activation)
    fig, axes = plt.subplots(1, n_activations, figsize=(4 * n_activations, 4))
    if n_activations == 1:
        axes = [axes]

    for idx, (activation, data) in enumerate(sorted(by_activation.items())):
        ax = axes[idx]

        # Get unique depths
        bottom_depths = sorted(set(int(s['bottom_depth']) for s in data))
        top_depths = sorted(set(int(s['top_depth']) for s in data))

        # Create grid
        grid = np.zeros((len(bottom_depths), len(top_depths)))

        for s in data:
            i = bottom_depths.index(int(s['bottom_depth']))
            j = top_depths.index(int(s['top_depth']))
            imp = s.get('improvement_mean')
            if imp is not None:
                grid[i, j] = imp * 100  # Convert to percentage

        # Plot heatmap
        im = ax.imshow(grid, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

        # Labels
        ax.set_xticks(range(len(top_depths)))
        ax.set_xticklabels([f'{d}' for d in top_depths])
        ax.set_yticks(range(len(bottom_depths)))
        ax.set_yticklabels([f'{d}' for d in bottom_depths])

        # Annotations
        for i in range(len(bottom_depths)):
            for j in range(len(top_depths)):
                val = grid[i, j]
                color = 'white' if abs(val) > 5 else 'black'
                ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                        color=color, fontsize=9)

        ax.set_xlabel('Top Depth')
        ax.set_ylabel('Bottom Depth')
        ax.set_title(ACTIVATION_LABELS.get(activation, activation))

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def create_transfer_matrix(
    transfer: List[Dict],
    output_path: str,
    title: str = "Transfer Learning: Source -> Target Benefit"
) -> None:
    """Matrix showing transfer benefit by source/target dataset."""
    if not transfer:
        print(f"   Skipping transfer matrix (no data)")
        return

    # Filter to train_all mode (full fine-tuning) for cleaner visualization
    transfer_trainall = [t for t in transfer if t.get('freeze_mode') == 'train_all']
    if not transfer_trainall:
        transfer_trainall = transfer

    # Get unique datasets
    datasets = sorted(set(
        list(set(t['source_dataset'] for t in transfer_trainall)) +
        list(set(t['target_dataset'] for t in transfer_trainall))
    ))

    # Group by activation
    by_activation = defaultdict(list)
    for t in transfer_trainall:
        by_activation[t['activation']].append(t)

    n_activations = len(by_activation)
    fig, axes = plt.subplots(1, n_activations, figsize=(4 * n_activations, 4))
    if n_activations == 1:
        axes = [axes]

    for idx, (activation, data) in enumerate(sorted(by_activation.items())):
        ax = axes[idx]

        # Create matrix
        grid = np.zeros((len(datasets), len(datasets)))
        grid[:] = np.nan  # Mark diagonal as N/A

        for t in data:
            source = t['source_dataset']
            target = t['target_dataset']
            benefit = t.get('transfer_benefit_mean')

            if source in datasets and target in datasets and benefit is not None:
                i = datasets.index(source)
                j = datasets.index(target)
                grid[i, j] = benefit * 100

        # Plot heatmap
        im = ax.imshow(grid, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

        # Labels
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)

        # Annotations
        for i in range(len(datasets)):
            for j in range(len(datasets)):
                val = grid[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 5 else 'black'
                    ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                            color=color, fontsize=9)

        ax.set_xlabel('Target Dataset')
        ax.set_ylabel('Source Dataset')
        ax.set_title(ACTIVATION_LABELS.get(activation, activation))

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def create_ensemble_scaling(
    ensemble: List[Dict],
    output_path: str,
    title: str = "Ensemble Scaling: Size vs Improvement"
) -> None:
    """Line chart showing ensemble improvement by size."""
    if not ensemble:
        print(f"   Skipping ensemble scaling (no data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: By ensemble type
    by_type = defaultdict(lambda: defaultdict(list))
    for e in ensemble:
        etype = e['ensemble_type']
        size = int(e['ensemble_size'])
        imp = e.get('improvement_mean')
        if imp is not None:
            by_type[etype][size].append(imp * 100)

    sizes = sorted(set(int(e['ensemble_size']) for e in ensemble))

    for etype in sorted(by_type.keys()):
        means = []
        for size in sizes:
            vals = by_type[etype].get(size, [0])
            means.append(np.mean(vals))

        color = ENSEMBLE_COLORS.get(etype, '#888888')
        ax1.plot(sizes, means, 'o-', color=color, label=etype.capitalize(),
                linewidth=2, markersize=8)

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Ensemble Size')
    ax1.set_ylabel('Improvement over Best Individual (%)')
    ax1.set_title('By Ensemble Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: By activation
    by_activation = defaultdict(lambda: defaultdict(list))
    for e in ensemble:
        activation = e['activation']
        size = int(e['ensemble_size'])
        imp = e.get('improvement_mean')
        if imp is not None:
            by_activation[activation][size].append(imp * 100)

    for activation in sorted(by_activation.keys()):
        means = []
        for size in sizes:
            vals = by_activation[activation].get(size, [0])
            means.append(np.mean(vals))

        color = ACTIVATION_COLORS.get(activation, '#888888')
        label = ACTIVATION_LABELS.get(activation, activation)
        ax2.plot(sizes, means, 'o-', color=color, label=label,
                linewidth=2, markersize=8)

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Ensemble Size')
    ax2.set_ylabel('Improvement over Best Individual (%)')
    ax2.set_title('By Activation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def create_composition_summary(
    stacking: List[Dict],
    transfer: List[Dict],
    ensemble: List[Dict],
    output_path: str,
    title: str = "Phase 8: Composition Summary"
) -> None:
    """Combined bar chart comparing all three experiment types."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute average improvements by activation for each type
    activations = ['relu', 'sigmoid', 'sine', 'tanh']
    x = np.arange(len(activations))
    width = 0.25

    # Stacking improvements
    stacking_by_act = defaultdict(list)
    for s in stacking:
        imp = s.get('improvement_mean')
        if imp is not None:
            stacking_by_act[s['activation']].append(imp * 100)

    stacking_means = [np.mean(stacking_by_act.get(a, [0])) for a in activations]

    # Transfer benefits
    transfer_by_act = defaultdict(list)
    for t in transfer:
        benefit = t.get('transfer_benefit_mean')
        if benefit is not None:
            transfer_by_act[t['activation']].append(benefit * 100)

    transfer_means = [np.mean(transfer_by_act.get(a, [0])) for a in activations]

    # Ensemble improvements
    ensemble_by_act = defaultdict(list)
    for e in ensemble:
        imp = e.get('improvement_mean')
        if imp is not None:
            ensemble_by_act[e['activation']].append(imp * 100)

    ensemble_means = [np.mean(ensemble_by_act.get(a, [0])) for a in activations]

    # Plot bars
    bars1 = ax.bar(x - width, stacking_means, width, label='Stacking', color='#3498db')
    bars2 = ax.bar(x, transfer_means, width, label='Transfer', color='#2ecc71')
    bars3 = ax.bar(x + width, ensemble_means, width, label='Ensemble', color='#f39c12')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Activation')
    ax.set_ylabel('Average Improvement (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([ACTIVATION_LABELS.get(a, a) for a in activations])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   Saved: {output_path}")


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    viz_path = project_root / 'visualizations'
    viz_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("PHASE 8 VISUALIZATIONS: Combination & Transfer")
    print("=" * 60)

    # Load data
    stacking = []
    transfer = []
    ensemble = []

    stacking_csv = data_path / 'phase8_stacking.csv'
    if stacking_csv.exists():
        stacking = load_csv(str(stacking_csv))
        print(f"\nLoaded {len(stacking)} stacking configurations")

    transfer_csv = data_path / 'phase8_transfer.csv'
    if transfer_csv.exists():
        transfer = load_csv(str(transfer_csv))
        print(f"Loaded {len(transfer)} transfer configurations")

    ensemble_csv = data_path / 'phase8_ensemble.csv'
    if ensemble_csv.exists():
        ensemble = load_csv(str(ensemble_csv))
        print(f"Loaded {len(ensemble)} ensemble configurations")

    if not stacking and not transfer and not ensemble:
        print("\nNo Phase 8 data found. Run aggregate_phase8.py first.")
        return

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Stacking heatmap
    create_stacking_heatmap(
        stacking,
        str(viz_path / 'phase8_stacking_heatmap.png')
    )

    # 2. Transfer matrix
    create_transfer_matrix(
        transfer,
        str(viz_path / 'phase8_transfer_matrix.png')
    )

    # 3. Ensemble scaling
    create_ensemble_scaling(
        ensemble,
        str(viz_path / 'phase8_ensemble_scaling.png')
    )

    # 4. Combined summary
    create_composition_summary(
        stacking, transfer, ensemble,
        str(viz_path / 'phase8_composition_summary.png')
    )

    print("\n" + "=" * 60)
    print("Phase 8 visualizations complete!")
    print(f"Charts saved to: {viz_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
