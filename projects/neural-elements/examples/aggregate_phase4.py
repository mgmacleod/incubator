#!/usr/bin/env python3
"""
Aggregate Phase 4 experiment results.

Run this after Phase 4 experiments complete to:
1. Aggregate results by configuration (including skip connections)
2. Export to CSV for external analysis
3. Print summary statistics comparing baseline vs skip connections
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
    get_summary_tables,
)


def print_phase4_summary(stats):
    """Print Phase 4 specific summary with skip connection comparison."""
    print("=" * 60)
    print("PHASE 4 RESULTS SUMMARY: Extended Depth Study")
    print("=" * 60)

    # Separate baseline and skip connection results
    baseline = [s for s in stats if not s.skip_connections]
    with_skip = [s for s in stats if s.skip_connections]

    print(f"\nTotal configurations: {len(stats)}")
    print(f"   Baseline (no skip): {len(baseline)}")
    print(f"   With skip connections: {len(with_skip)}")

    # Filter to Phase 4 depths (6, 7, 8, 10)
    phase4_depths = {6, 7, 8, 10}
    baseline_p4 = [s for s in baseline if s.depth in phase4_depths]
    with_skip_p4 = [s for s in with_skip if s.depth in phase4_depths]

    print(f"\n   Phase 4 depths only:")
    print(f"      Baseline: {len(baseline_p4)}")
    print(f"      With skip: {len(with_skip_p4)}")

    # Activation ranking - baseline
    if baseline_p4:
        print("\n   Activation Ranking (Baseline, depths 6-10):")
        activation_means = {}
        for s in baseline_p4:
            activation_means.setdefault(s.activation, []).append(s.mean_accuracy)
        activation_avg = {
            act: sum(accs) / len(accs) * 100
            for act, accs in activation_means.items()
        }
        for i, (act, avg) in enumerate(sorted(activation_avg.items(), key=lambda x: -x[1]), 1):
            print(f"      {i}. {act:12s} {avg:.1f}%")

    # Skip connection effect by activation
    if baseline_p4 and with_skip_p4:
        print("\n   Skip Connection Effect by Activation:")
        for activation in sorted(set(s.activation for s in baseline_p4)):
            base_accs = [s.mean_accuracy for s in baseline_p4 if s.activation == activation]
            skip_accs = [s.mean_accuracy for s in with_skip_p4 if s.activation == activation]
            if base_accs and skip_accs:
                base_mean = sum(base_accs) / len(base_accs) * 100
                skip_mean = sum(skip_accs) / len(skip_accs) * 100
                delta = skip_mean - base_mean
                arrow = "" if delta > 0 else "" if delta < 0 else "="
                print(f"      {activation:12s} {base_mean:.1f}% -> {skip_mean:.1f}% ({arrow}{delta:+.1f}%)")

    # Depth effect - sigmoid specifically (the collapse question)
    print("\n   Sigmoid Depth Collapse (Key Question):")
    sigmoid_baseline = [s for s in baseline if s.activation == 'sigmoid']
    sigmoid_baseline.sort(key=lambda s: s.depth)
    for s in sigmoid_baseline:
        bar = "" * int(s.mean_accuracy * 20)
        print(f"      Depth {s.depth:2d}: {s.mean_accuracy*100:5.1f}% [{s.ci_lower*100:.1f}%, {s.ci_upper*100:.1f}%] {bar}")

    if with_skip:
        sigmoid_skip = [s for s in with_skip if s.activation == 'sigmoid']
        sigmoid_skip.sort(key=lambda s: s.depth)
        print("\n   Sigmoid with Skip Connections:")
        for s in sigmoid_skip:
            bar = "" * int(s.mean_accuracy * 20)
            print(f"      Depth {s.depth:2d}: {s.mean_accuracy*100:5.1f}% [{s.ci_lower*100:.1f}%, {s.ci_upper*100:.1f}%] {bar}")

    # Sine stability
    print("\n   Sine Stability (Key Question):")
    sine_baseline = [s for s in baseline if s.activation == 'sine']
    sine_baseline.sort(key=lambda s: s.depth)
    for s in sine_baseline:
        bar = "" * int(s.mean_accuracy * 20)
        print(f"      Depth {s.depth:2d}: {s.mean_accuracy*100:5.1f}% [{s.ci_lower*100:.1f}%, {s.ci_upper*100:.1f}%] {bar}")

    # Best configurations
    print("\n   Best by Dataset (Baseline):")
    for dataset in ['xor', 'moons', 'circles', 'spirals']:
        dataset_stats = [s for s in baseline_p4 if s.dataset == dataset]
        if dataset_stats:
            best = max(dataset_stats, key=lambda s: s.mean_accuracy)
            skip_suffix = " (skip)" if best.skip_connections else ""
            print(f"      {dataset:10s} -> {best.activation} d={best.depth}{skip_suffix} "
                  f"{best.mean_accuracy*100:.1f}%")

    print("=" * 60)


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    output_dir = project_root / 'data'

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
        print("   Run examples/run_phase4_training.py first.")
        return

    print(f"\n   Found {completed} completed experiments")

    # Aggregate results
    print("\n   Aggregating results...")
    aggregated = aggregate_experiments(store, status='completed')
    print(f"   Generated {len(aggregated)} configuration summaries")

    # Filter to Phase 4 depths
    phase4_depths = {6, 7, 8, 10}
    phase4_stats = [s for s in aggregated if s.depth in phase4_depths]

    if not phase4_stats:
        print("\n   No Phase 4 experiments found (depths 6, 7, 8, 10).")
        print("   The aggregated results may be from other phases.")

        # Still export all results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = output_dir / f'all_experiments_{timestamp}.csv'
        export_to_csv(aggregated, str(csv_path))
        print(f"\n   Exported all {len(aggregated)} configs to: {csv_path}")
        return

    # Export Phase 4 results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'phase4_summary_{timestamp}.csv'
    export_to_csv(phase4_stats, str(csv_path))
    print(f"\n   Exported Phase 4 results to: {csv_path}")

    # Also create a latest version
    latest_path = output_dir / 'phase4_summary.csv'
    export_to_csv(phase4_stats, str(latest_path))
    print(f"   Also saved as: {latest_path}")

    # Also export ALL results (including previous phases) for combined analysis
    all_path = output_dir / 'all_phases_summary.csv'
    export_to_csv(aggregated, str(all_path))
    print(f"   Combined all phases: {all_path}")

    # Print summary
    print("\n")
    print_phase4_summary(aggregated)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review phase4_summary.csv for detailed results")
    print("  2. Run examples/visualize_phase4.py for charts")
    print("  3. Compare with Phase 3 results in all_phases_summary.csv")
    print("=" * 60)


if __name__ == '__main__':
    main()
