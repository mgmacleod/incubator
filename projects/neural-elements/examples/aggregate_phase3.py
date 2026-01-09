#!/usr/bin/env python3
"""
Aggregate Phase 3 experiment results.

Run this after Phase 3 experiments complete to:
1. Aggregate results by configuration
2. Export to CSV for external analysis
3. Print summary statistics
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
    print_summary,
)


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    output_dir = project_root / 'data'

    # Initialize store
    store = ExperimentStore(str(data_path))

    # Get statistics to see what we have
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
        print("\n❌ No completed experiments found.")
        print("   Run examples/run_phase3_training.py first.")
        return

    print(f"\n✅ Found {completed} completed experiments")

    # Aggregate results
    print("\n📊 Aggregating results...")
    aggregated = aggregate_experiments(store, status='completed')
    print(f"   Generated {len(aggregated)} configuration summaries")

    # Export to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'phase3_summary_{timestamp}.csv'
    export_to_csv(aggregated, str(csv_path))
    print(f"\n💾 Exported to: {csv_path}")

    # Also create a latest symlink/copy
    latest_path = output_dir / 'phase3_summary.csv'
    export_to_csv(aggregated, str(latest_path))
    print(f"   Also saved as: {latest_path}")

    # Print summary
    print("\n")
    print_summary(aggregated)

    # Show sample of results
    print("\n📋 Sample configurations (top 10 by accuracy):")
    top_10 = sorted(aggregated, key=lambda s: s.mean_accuracy, reverse=True)[:10]
    for i, stat in enumerate(top_10, 1):
        print(f"   {i:2d}. {stat.activation:12s} d={stat.depth} {stat.dataset:10s} "
              f"{stat.mean_accuracy*100:.1f}% ± {stat.std_accuracy*100:.1f}% "
              f"(n={stat.n_completed})")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review phase3_summary.csv for detailed results")
    print("  2. Run examples/visualize_phase3.py for charts")
    print("=" * 60)


if __name__ == '__main__':
    main()
