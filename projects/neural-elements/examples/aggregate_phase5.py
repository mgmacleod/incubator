#!/usr/bin/env python3
"""
Aggregate Phase 5 experiment results.

Run this after Phase 5 experiments complete to:
1. Aggregate results by architecture pattern
2. Export to CSV for external analysis
3. Print summary statistics comparing architectures
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.persistence import ExperimentStore
from src.analysis import (
    aggregate_experiments,
    export_to_csv,
    Phase5Config,
)


def classify_architecture(hidden_layers):
    """Classify a hidden_layers list into an architecture pattern."""
    if not hidden_layers:
        return 'linear'

    if all(w == hidden_layers[0] for w in hidden_layers):
        return 'uniform'

    # Check for bottleneck patterns (min in middle)
    if len(hidden_layers) >= 3:
        middle = len(hidden_layers) // 2
        if hidden_layers[middle] < hidden_layers[0] and hidden_layers[middle] < hidden_layers[-1]:
            return 'bottleneck'

    # Check for pyramid patterns
    if hidden_layers == sorted(hidden_layers):
        return 'expanding'
    if hidden_layers == sorted(hidden_layers, reverse=True):
        return 'contracting'

    # Diamond pattern: expands then contracts
    peak_idx = hidden_layers.index(max(hidden_layers))
    if 0 < peak_idx < len(hidden_layers) - 1:
        before = hidden_layers[:peak_idx]
        after = hidden_layers[peak_idx + 1:]
        if before == sorted(before) and after == sorted(after, reverse=True):
            return 'diamond'

    return 'mixed'


def print_phase5_summary(stats, config):
    """Print Phase 5 specific summary with architecture comparisons."""
    print("=" * 60)
    print("PHASE 5 RESULTS SUMMARY: Architecture Space Exploration")
    print("=" * 60)

    print(f"\nTotal configurations: {len(stats)}")

    # Group by architecture pattern
    pattern_groups = {}
    for s in stats:
        # Parse hidden_layers from the element name or use width/depth as proxy
        # For Phase 5, we stored hidden_layers in the element name format
        pattern = 'unknown'

        # Try to reconstruct hidden_layers from element name parsing
        # The aggregator returns width (max width) and depth
        # For non-uniform, we need to check against our known architectures
        for arch_name, arch_layers in config.architectures.items():
            # Check if this matches
            if s.depth == len(arch_layers):
                # Check if width matches max width
                if s.width == max(arch_layers):
                    pattern = classify_architecture(arch_layers)
                    break

        pattern_groups.setdefault(pattern, []).append(s)

    print("\n   By Architecture Pattern:")
    for pattern, group in sorted(pattern_groups.items()):
        accs = [s.mean_accuracy for s in group]
        avg = sum(accs) / len(accs) * 100 if accs else 0
        print(f"      {pattern:15s}: {len(group):3d} configs, avg {avg:.1f}%")

    # Activation ranking
    print("\n   Activation Ranking (Overall):")
    activation_means = {}
    for s in stats:
        activation_means.setdefault(s.activation, []).append(s.mean_accuracy)
    activation_avg = {
        act: sum(accs) / len(accs) * 100
        for act, accs in activation_means.items()
    }
    for i, (act, avg) in enumerate(sorted(activation_avg.items(), key=lambda x: -x[1]), 1):
        print(f"      {i}. {act:12s} {avg:.1f}%")

    # Bottleneck analysis - the key question
    print("\n   Bottleneck Architecture Analysis:")
    bottleneck_configs = [
        ('bottleneck_severe', [32, 8, 32]),
        ('bottleneck_moderate', [16, 4, 16]),
        ('bottleneck_extreme', [8, 2, 8]),
    ]

    for name, layers in bottleneck_configs:
        depth = len(layers)
        max_width = max(layers)
        min_width = min(layers)
        # Find matching stats
        matches = [s for s in stats if s.depth == depth and s.width == max_width]
        if matches:
            avg_acc = sum(s.mean_accuracy for s in matches) / len(matches) * 100
            print(f"      {name:20s} {str(layers):15s}: {avg_acc:.1f}%")
            if min_width == 2:
                print(f"         ^ Width-2 bottleneck - Phase 3 showed this is devastating")
        else:
            print(f"      {name:20s} {str(layers):15s}: (no data)")

    # Pyramid analysis
    print("\n   Pyramid Architecture Analysis:")
    pyramid_configs = [
        ('expanding', [4, 8, 16]),
        ('contracting', [16, 8, 4]),
        ('diamond', [4, 8, 16, 8, 4]),
    ]

    for name, layers in pyramid_configs:
        depth = len(layers)
        max_width = max(layers)
        matches = [s for s in stats if s.depth == depth and s.width == max_width]
        if matches:
            avg_acc = sum(s.mean_accuracy for s in matches) / len(matches) * 100
            print(f"      {name:15s} {str(layers):15s}: {avg_acc:.1f}%")
        else:
            print(f"      {name:15s} {str(layers):15s}: (no data)")

    # Parameter efficiency comparison
    print("\n   Parameter Efficiency (same-ish param count):")
    param_configs = [
        ('wide_shallow', [32], 32 * 2 + 32 * 1 + 1 + 32),  # ~97 params
        ('medium', [12, 12], 12 * 2 + 12 * 12 + 12 * 1 + 12 + 12 + 1),  # ~193 params
        ('narrow_deep', [8, 8, 8], 8 * 2 + 8 * 8 + 8 * 8 + 8 * 1 + 8 + 8 + 8 + 1),  # ~169 params
    ]

    for name, layers, params in param_configs:
        depth = len(layers)
        max_width = max(layers)
        matches = [s for s in stats if s.depth == depth and s.width == max_width]
        if matches:
            avg_acc = sum(s.mean_accuracy for s in matches) / len(matches) * 100
            print(f"      {name:15s} {str(layers):15s} (~{params:3d} params): {avg_acc:.1f}%")
        else:
            print(f"      {name:15s} {str(layers):15s} (~{params:3d} params): (no data)")

    # Best by dataset
    print("\n   Best Architecture by Dataset:")
    for dataset in ['xor', 'moons', 'circles', 'spirals']:
        dataset_stats = [s for s in stats if s.dataset == dataset]
        if dataset_stats:
            best = max(dataset_stats, key=lambda s: s.mean_accuracy)
            print(f"      {dataset:10s} -> {best.activation} d={best.depth} w={best.width} "
                  f"{best.mean_accuracy*100:.1f}%")

    print("=" * 60)


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    output_dir = project_root / 'data'

    # Load config for reference
    config = Phase5Config()

    # Initialize store
    store = ExperimentStore(str(data_path))

    # Get statistics
    stats = store.get_statistics()
    print("=" * 60)
    print("EXPERIMENT STORE STATUS")
    print("=" * 60)
    print(f"\nTotal experiments: {stats['total']}")
    print(f"\nBy status:")
    for status, count in sorted(stats['by_status'].items()):
        print(f"   {status}: {count}")

    # Check for completed experiments
    completed = stats['by_status'].get('completed', 0)
    if completed == 0:
        print("\n   No completed experiments found.")
        print("   Run examples/run_phase5_training.py first.")
        return

    print(f"\n   Found {completed} completed experiments")

    # Aggregate results
    print("\n   Aggregating results...")
    aggregated = aggregate_experiments(store, status='completed')
    print(f"   Generated {len(aggregated)} configuration summaries")

    # Filter to Phase 5 architectures (non-uniform or specific patterns)
    # Phase 5 architectures have specific patterns we can identify
    phase5_archs = list(config.architectures.values())
    phase5_depths = set(len(a) for a in phase5_archs)
    phase5_widths = set(max(a) for a in phase5_archs)

    # Filter to Phase 5 configs by checking depth/width combinations
    phase5_stats = [s for s in aggregated
                    if s.depth in phase5_depths and s.width in phase5_widths]

    if not phase5_stats:
        print("\n   No Phase 5 experiments found matching expected architectures.")
        print("   The aggregated results may be from other phases.")

        # Still export all results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = output_dir / f'all_experiments_{timestamp}.csv'
        export_to_csv(aggregated, str(csv_path))
        print(f"\n   Exported all {len(aggregated)} configs to: {csv_path}")
        return

    # Export Phase 5 results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'phase5_summary_{timestamp}.csv'
    export_to_csv(phase5_stats, str(csv_path))
    print(f"\n   Exported Phase 5 results to: {csv_path}")

    # Also create a latest version
    latest_path = output_dir / 'phase5_summary.csv'
    export_to_csv(phase5_stats, str(latest_path))
    print(f"   Also saved as: {latest_path}")

    # Also export ALL results (including previous phases) for combined analysis
    all_path = output_dir / 'all_phases_summary.csv'
    export_to_csv(aggregated, str(all_path))
    print(f"   Combined all phases: {all_path}")

    # Print summary
    print("\n")
    print_phase5_summary(phase5_stats, config)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review phase5_summary.csv for detailed results")
    print("  2. Run examples/visualize_phase5.py for charts")
    print("  3. Compare with previous phases in all_phases_summary.csv")
    print("=" * 60)


if __name__ == '__main__':
    main()
