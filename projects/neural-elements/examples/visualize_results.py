#!/usr/bin/env python3
"""
Visualize Neural Elements experiment results.

Creates heatmaps and charts showing patterns across the "periodic table".
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from statistics import mean, stdev


def load_results():
    """Load all experiment results from index."""
    index_path = Path(__file__).parent.parent / 'data' / 'index.json'
    index = json.loads(index_path.read_text())
    return index['experiments']


def parse_element(name):
    """Parse element name into (activation, depth, width)."""
    parts = name.split('-')
    if len(parts) != 2:
        return None, None, None
    act = parts[0]
    dims = parts[1].split('x')
    if len(dims) != 2:
        return None, None, None
    return act, int(dims[0]), int(dims[1])


def create_periodic_table_heatmap(experiments, output_path='periodic_table_heatmap.png'):
    """
    Create a heatmap visualization of the periodic table.
    Rows = depth, Columns = activation, Color = accuracy
    """
    # Collect results
    results = defaultdict(lambda: defaultdict(list))

    for exp in experiments.values():
        if exp.get('status') != 'completed' or exp.get('final_accuracy') is None:
            continue
        name = exp['element_name']
        act, depth, width = parse_element(name)
        if act and depth and width == 4:  # Standard width for main table
            results[act][depth].append(exp['final_accuracy'])

    # Define order
    activations = ['LIN', 'SIG', 'REL', 'LEA', 'TAN', 'GEL', 'SIN', 'SWI']
    act_labels = ['Linear', 'Sigmoid', 'ReLU', 'Leaky\nReLU', 'Tanh', 'GELU', 'Sine', 'Swish']
    depths = [1, 2, 3, 4, 5]

    # Build matrix
    matrix = np.zeros((len(depths), len(activations)))
    matrix[:] = np.nan

    for i, depth in enumerate(depths):
        for j, act in enumerate(activations):
            accs = results[act].get(depth, [])
            if accs:
                matrix[i, j] = mean(accs)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom colormap: red (bad) -> yellow -> green (good)
    cmap = plt.cm.RdYlGn

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0.5, vmax=1.0)

    # Add text annotations
    for i in range(len(depths)):
        for j in range(len(activations)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.65 or val > 0.9 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=11, fontweight='bold')

    # Labels
    ax.set_xticks(range(len(activations)))
    ax.set_xticklabels(act_labels, fontsize=10)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f'{d} Layer{"s" if d > 1 else ""}' for d in depths], fontsize=10)

    ax.set_xlabel('Activation Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Network Depth', fontsize=12, fontweight='bold')
    ax.set_title('Neural Elements Periodic Table\n(Average Accuracy, Width=4)',
                fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy', fontsize=11)

    # Grid
    ax.set_xticks(np.arange(-0.5, len(activations), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(depths), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_depth_effect_chart(experiments, output_path='depth_effect.png'):
    """Show how different activations respond to depth."""

    # Collect results
    results = defaultdict(lambda: defaultdict(list))

    for exp in experiments.values():
        if exp.get('status') != 'completed' or exp.get('final_accuracy') is None:
            continue
        name = exp['element_name']
        act, depth, width = parse_element(name)
        if act and depth and width >= 4:  # Exclude narrow
            results[act][depth].append(exp['final_accuracy'])

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'SIG': '#e74c3c',  # Red - the villain
        'REL': '#3498db',  # Blue
        'TAN': '#2ecc71',  # Green
        'SIN': '#9b59b6',  # Purple - the hero
        'GEL': '#f39c12',  # Orange
        'LEA': '#1abc9c',  # Teal
    }

    depths = [1, 2, 3, 4, 5]

    for act in ['SIG', 'REL', 'TAN', 'GEL', 'SIN', 'LEA']:
        accs = []
        for d in depths:
            vals = results[act].get(d, [])
            accs.append(mean(vals) if vals else np.nan)

        ax.plot(depths, accs, 'o-', color=colors[act], linewidth=2.5,
               markersize=8, label=act)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')

    ax.set_xlabel('Network Depth (layers)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Depth Effect by Activation Function\n(Sigmoid collapse vs Sine resilience)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(depths)
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.annotate('Sigmoid\ncollapses!', xy=(5, 0.55), xytext=(4.5, 0.65),
               fontsize=10, color='#e74c3c',
               arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    ax.annotate('Sine stays\nstrong', xy=(5, 0.82), xytext=(4, 0.92),
               fontsize=10, color='#9b59b6',
               arrowprops=dict(arrowstyle='->', color='#9b59b6'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_dataset_comparison(experiments, output_path='dataset_comparison.png'):
    """Compare activation performance across datasets."""

    results = defaultdict(lambda: defaultdict(list))

    for exp in experiments.values():
        if exp.get('status') != 'completed' or exp.get('final_accuracy') is None:
            continue
        name = exp['element_name']
        act, depth, width = parse_element(name)
        dataset = exp.get('dataset', 'unknown')
        if act:
            results[dataset][act].append(exp['final_accuracy'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    datasets = ['xor', 'moons', 'circles', 'spirals']
    titles = ['XOR (Classic Test)', 'Moons (Curved Boundary)',
              'Circles (Radial)', 'Spirals (Hard)']

    activations = ['REL', 'TAN', 'GEL', 'SIG', 'SIN', 'LEA', 'SWI', 'LIN']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']

    for ax, dataset, title in zip(axes.flat, datasets, titles):
        accs = []
        for act in activations:
            vals = results[dataset].get(act, [])
            accs.append(mean(vals) if vals else 0)

        bars = ax.bar(activations, accs, color=colors)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Accuracy')
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Highlight best
        max_idx = np.argmax(accs)
        bars[max_idx].set_edgecolor('black')
        bars[max_idx].set_linewidth(2)

        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                       f'{acc:.2f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Performance by Dataset and Activation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def create_width_effect_chart(experiments, output_path='width_effect.png'):
    """Show how width affects performance."""

    results = defaultdict(lambda: defaultdict(list))

    for exp in experiments.values():
        if exp.get('status') != 'completed' or exp.get('final_accuracy') is None:
            continue
        name = exp['element_name']
        act, depth, width = parse_element(name)
        if act == 'REL':  # Focus on ReLU
            results[depth][width].append(exp['final_accuracy'])

    fig, ax = plt.subplots(figsize=(10, 6))

    widths = [2, 4, 8, 16]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

    x = np.arange(len([1, 2, 3]))
    bar_width = 0.2

    for i, width in enumerate(widths):
        accs = []
        for depth in [1, 2, 3]:
            vals = results[depth].get(width, [])
            accs.append(mean(vals) if vals else 0)

        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, accs, bar_width, label=f'Width={width}', color=colors[i])

    ax.set_xlabel('Network Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Width vs Depth Trade-off (ReLU)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['1 Layer', '2 Layers', '3 Layers'])
    ax.set_ylim(0.6, 1.0)
    ax.legend(title='Width')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def main():
    print("Loading experiment results...")
    experiments = load_results()
    print(f"Loaded {len(experiments)} experiments")

    output_dir = Path(__file__).parent.parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")

    create_periodic_table_heatmap(
        experiments,
        output_dir / 'periodic_table_heatmap.png'
    )

    create_depth_effect_chart(
        experiments,
        output_dir / 'depth_effect.png'
    )

    create_dataset_comparison(
        experiments,
        output_dir / 'dataset_comparison.png'
    )

    create_width_effect_chart(
        experiments,
        output_dir / 'width_effect.png'
    )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
