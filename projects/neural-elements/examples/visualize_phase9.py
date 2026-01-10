#!/usr/bin/env python3
"""
Phase 9: Evolutionary Discovery - Visualization and Analysis

Generates visualizations and analysis from evolution checkpoints:
- Fitness progression over generations
- Architecture distribution analysis
- Champion comparison with Phase 3-8 baselines
- Diversity metrics over time

Usage:
    python examples/visualize_phase9.py [options]

Options:
    --checkpoint PATH   Specific checkpoint file to analyze
    --latest            Analyze most recent checkpoint (default)
    --all               Analyze all checkpoints
    --output DIR        Output directory for plots (default: data/evolution/plots)
    --no-plots          Skip plot generation, just print summary
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evolution.checkpoint import EvolutionCheckpoint
from src.evolution.genome import Genome
from src.evolution.fitness import weighted_fitness


def find_checkpoints(data_dir: Path) -> list:
    """Find all checkpoint files, sorted by modification time."""
    evolution_dir = data_dir / 'evolution'
    if not evolution_dir.exists():
        return []

    checkpoints = list(evolution_dir.glob('*.json'))
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints


def load_checkpoint(path: Path) -> EvolutionCheckpoint:
    """Load a checkpoint file."""
    return EvolutionCheckpoint.load(path)


def print_checkpoint_summary(checkpoint: EvolutionCheckpoint, path: Path):
    """Print summary of a checkpoint."""
    print(f"\n{'='*60}")
    print(f"Checkpoint: {path.name}")
    print(f"{'='*60}")

    print(f"\nRun ID:      {checkpoint.run_id}")
    print(f"Dataset:     {checkpoint.dataset_name}")
    print(f"Generation:  {checkpoint.generation}")
    print(f"Evaluations: {checkpoint.total_evaluations}")
    print(f"Timestamp:   {checkpoint.timestamp}")

    # Population stats
    population = checkpoint.get_population()
    print(f"\nPopulation size: {len(population)}")

    # Activation distribution
    activation_counts = defaultdict(int)
    depth_counts = defaultdict(int)
    skip_count = 0

    for genome in population:
        for act in genome.activations:
            activation_counts[act] += 1
        depth_counts[genome.depth] += 1
        if genome.skip_connections:
            skip_count += 1

    print(f"\nActivation distribution:")
    for act, count in sorted(activation_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / sum(activation_counts.values())
        print(f"  {act:12s}: {count:3d} ({pct:5.1f}%)")

    print(f"\nDepth distribution:")
    for depth, count in sorted(depth_counts.items()):
        pct = 100 * count / len(population)
        print(f"  {depth} layers: {count:3d} ({pct:5.1f}%)")

    print(f"\nSkip connections: {skip_count}/{len(population)} ({100*skip_count/len(population):.1f}%)")

    # Champions
    champions = checkpoint.get_champions()
    if champions:
        print(f"\nTop Champions ({len(champions)} total):")
        print(f"  {'Rank':<5} {'Architecture':<30} {'Accuracy':<10} {'Params':<8} {'Fitness':<8}")
        print(f"  {'-'*5} {'-'*30} {'-'*10} {'-'*8} {'-'*8}")

        for i, champ in enumerate(champions[:10], 1):
            if champ.fitness:
                acc = champ.fitness.test_accuracy
                fit = weighted_fitness(champ.fitness)
            else:
                acc = fit = 0
            print(f"  {i:<5} {champ.architecture_string:<30} {acc:<10.4f} {champ.total_params:<8} {fit:<8.4f}")

    # Fitness history
    history = checkpoint.history
    if 'fitness_trajectory' in history and history['fitness_trajectory']:
        trajectory = history['fitness_trajectory']
        print(f"\nFitness progression:")
        print(f"  Initial:  {trajectory[0]:.4f}")
        print(f"  Final:    {trajectory[-1]:.4f}")
        print(f"  Best:     {max(trajectory):.4f}")
        print(f"  Improvement: {trajectory[-1] - trajectory[0]:+.4f}")


def analyze_champions_vs_baselines(champions: list):
    """Compare champions against known Phase 3-8 baselines."""
    print(f"\n{'='*60}")
    print("Champion Analysis vs Phase 3-8 Baselines")
    print(f"{'='*60}")

    # Known good baselines from earlier phases
    baselines = {
        'relu[8-8-8]': 0.95,
        'leaky_relu[8-8-8]': 0.96,
        'sine[8-8-8]': 0.94,
        'tanh[8-8-8]': 0.95,
        'gelu[8]': 0.92,
    }

    print(f"\nBaseline accuracies (Phase 3-8):")
    for arch, acc in baselines.items():
        print(f"  {arch:<25}: {acc:.2f}")

    if not champions:
        print("\nNo champions to compare.")
        return

    best_champion = max(champions, key=lambda g: g.fitness.test_accuracy if g.fitness else 0)
    best_acc = best_champion.fitness.test_accuracy if best_champion.fitness else 0
    best_baseline_acc = max(baselines.values())

    print(f"\nBest evolved architecture:")
    print(f"  {best_champion.architecture_string}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  Params:   {best_champion.total_params}")

    if best_acc > best_baseline_acc:
        print(f"\n  IMPROVEMENT over baselines: +{best_acc - best_baseline_acc:.4f}")
    else:
        print(f"\n  Difference from best baseline: {best_acc - best_baseline_acc:+.4f}")

    # Novel architectures (not in baselines)
    novel = []
    for champ in champions:
        arch = champ.architecture_string
        is_novel = True
        for baseline in baselines:
            # Simple check - exact match or similar pattern
            if baseline.replace('[', '').replace(']', '').replace('-', '') in \
               arch.replace('[', '').replace(']', '').replace('-', ''):
                is_novel = False
                break
        if is_novel:
            novel.append(champ)

    if novel:
        print(f"\nNovel architectures discovered ({len(novel)}):")
        for champ in novel[:5]:
            acc = champ.fitness.test_accuracy if champ.fitness else 0
            print(f"  {champ.architecture_string:<30} | Acc: {acc:.3f} | Params: {champ.total_params}")


def plot_fitness_progression(checkpoint: EvolutionCheckpoint, output_dir: Path):
    """Generate fitness progression plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    history = checkpoint.history
    if 'fitness_trajectory' not in history or not history['fitness_trajectory']:
        print("No fitness trajectory data to plot")
        return

    trajectory = history['fitness_trajectory']
    generations = list(range(1, len(trajectory) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, trajectory, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title(f'Evolution Progress - {checkpoint.dataset_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(trajectory))

    # Add improvement annotation
    improvement = trajectory[-1] - trajectory[0]
    ax.annotate(
        f'Improvement: {improvement:+.4f}',
        xy=(len(trajectory), trajectory[-1]),
        xytext=(len(trajectory) * 0.7, trajectory[0] + improvement * 0.3),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{checkpoint.run_id}_fitness.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fitness plot: {output_path}")


def plot_architecture_distribution(checkpoint: EvolutionCheckpoint, output_dir: Path):
    """Generate architecture distribution plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    population = checkpoint.get_population()

    # Activation distribution
    activation_counts = defaultdict(int)
    for genome in population:
        primary = genome.activations[0]
        activation_counts[primary] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Activation pie chart
    ax1 = axes[0]
    labels = list(activation_counts.keys())
    sizes = list(activation_counts.values())
    colors = plt.cm.Set3(range(len(labels)))
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    ax1.set_title('Activation Distribution')

    # Depth histogram
    ax2 = axes[1]
    depths = [g.depth for g in population]
    ax2.hist(depths, bins=range(1, max(depths) + 2), edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Depth (layers)')
    ax2.set_ylabel('Count')
    ax2.set_title('Depth Distribution')

    plt.suptitle(f'Population Architecture - Gen {checkpoint.generation}', fontsize=14)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{checkpoint.run_id}_architecture.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved architecture plot: {output_path}")


def plot_diversity_over_time(checkpoint: EvolutionCheckpoint, output_dir: Path):
    """Generate diversity metrics plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    history = checkpoint.history
    if 'diversity_trajectory' not in history or not history['diversity_trajectory']:
        return

    diversity = history['diversity_trajectory']
    generations = list(range(1, len(diversity) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, diversity, 'g-', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Diversity (unique architectures ratio)', fontsize=12)
    ax.set_title(f'Population Diversity - {checkpoint.dataset_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(diversity))
    ax.set_ylim(0, 1.05)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{checkpoint.run_id}_diversity.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved diversity plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Phase 9 evolution results'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Specific checkpoint file to analyze'
    )
    parser.add_argument(
        '--latest', action='store_true', default=True,
        help='Analyze most recent checkpoint (default)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Analyze all checkpoints'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip plot generation'
    )
    args = parser.parse_args()

    data_dir = project_root / 'data'
    output_dir = Path(args.output) if args.output else data_dir / 'evolution' / 'plots'

    print("=" * 60)
    print("   NEURAL ELEMENTS - Phase 9 Analysis")
    print("=" * 60)

    # Find checkpoints
    if args.checkpoint:
        checkpoint_files = [Path(args.checkpoint)]
    else:
        checkpoint_files = find_checkpoints(data_dir)
        if not checkpoint_files:
            print("\nNo checkpoint files found in data/evolution/")
            print("Run evolution first: python examples/run_phase9_evolution.py")
            return

        if not args.all:
            checkpoint_files = checkpoint_files[:1]  # Latest only

    print(f"\nFound {len(checkpoint_files)} checkpoint(s)")

    # Analyze each checkpoint
    all_champions = []
    for cp_path in checkpoint_files:
        try:
            checkpoint = load_checkpoint(cp_path)
            print_checkpoint_summary(checkpoint, cp_path)

            champions = checkpoint.get_champions()
            all_champions.extend(champions)

            if not args.no_plots:
                plot_fitness_progression(checkpoint, output_dir)
                plot_architecture_distribution(checkpoint, output_dir)
                plot_diversity_over_time(checkpoint, output_dir)

        except Exception as e:
            print(f"\nError loading {cp_path}: {e}")

    # Cross-checkpoint analysis
    if all_champions:
        analyze_champions_vs_baselines(all_champions)

    if not args.no_plots:
        print(f"\nPlots saved to: {output_dir}")


if __name__ == '__main__':
    main()
