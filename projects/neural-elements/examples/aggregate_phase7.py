#!/usr/bin/env python3
"""
Aggregate Phase 7 experiment results with generalization metrics.

Run this after Phase 7 experiments complete to:
1. Aggregate results by configuration
2. Compute generalization metrics (gap, noise robustness)
3. Export to CSV for external analysis
4. Print summary with generalization insights
"""

import sys
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.persistence import ExperimentStore
from src.analysis.config_selection import Phase7Config

import numpy as np


def get_config_key(element_name, dataset, sample_size):
    """Generate a unique key for grouping trials."""
    return f"{element_name}_{dataset}_n{sample_size}"


def aggregate_phase7_experiments(store, status='completed'):
    """
    Aggregate Phase 7 experiments with generalization metrics.

    Returns list of dicts with accuracy stats and generalization metrics.
    """
    # Get all completed experiments
    experiments = store.list_experiments(status=status, limit=20000)

    # Group by configuration
    config_groups = defaultdict(list)
    for exp in experiments:
        # Load full result to get generalization metrics
        _, result = store.load_experiment(exp['experiment_id'])
        if result is None:
            continue

        # Skip experiments without Phase 7 fields
        if result.sample_size is None:
            continue

        # Load metadata for config details
        metadata = store.load_metadata(exp['experiment_id'])
        if metadata is None:
            continue

        config_key = get_config_key(
            exp['element_name'],
            exp.get('dataset', 'unknown'),
            result.sample_size
        )

        # Store both result and metadata
        config_groups[config_key].append({
            'result': result,
            'metadata': metadata,
            'experiment_id': exp['experiment_id'],
        })

    # Aggregate each configuration
    aggregated = []
    for config_key, trials in config_groups.items():
        if not trials:
            continue

        # Get config details from first trial
        first_result = trials[0]['result']
        first_metadata = trials[0]['metadata']
        element_config = first_metadata.element_config
        hidden_layers = element_config.get('hidden_layers', [])

        # Basic accuracy stats
        final_accs = [t['result'].final_accuracy for t in trials]
        train_accs = [t['result'].train_accuracy for t in trials if t['result'].train_accuracy is not None]
        test_accs = [t['result'].test_accuracy for t in trials if t['result'].test_accuracy is not None]
        gaps = [t['result'].generalization_gap for t in trials if t['result'].generalization_gap is not None]
        training_times = [t['result'].training_time_seconds for t in trials]

        # Aggregate noise robustness by noise level
        noise_by_level = defaultdict(list)
        for trial in trials:
            noise_dict = trial['result'].noise_robustness
            if noise_dict:
                for level, acc in noise_dict.items():
                    noise_by_level[level].append(acc)

        # Compute noise robustness aggregates
        noise_robustness_agg = {}
        for level, accs in sorted(noise_by_level.items()):
            noise_robustness_agg[f'noise_{level}_mean'] = float(np.mean(accs))
            noise_robustness_agg[f'noise_{level}_std'] = float(np.std(accs))

        # Compute noise drop (accuracy loss from clean to max noise)
        clean_acc = noise_by_level.get('0.0', noise_by_level.get('0', []))
        max_noise_key = max(noise_by_level.keys(), key=float) if noise_by_level else '0'
        noisy_acc = noise_by_level.get(max_noise_key, [])
        noise_drop = float(np.mean(clean_acc) - np.mean(noisy_acc)) if clean_acc and noisy_acc else 0.0

        aggregated.append({
            'config_key': config_key,
            'element_name': first_result.element_name,
            'activation': element_config.get('activation', 'unknown'),
            'hidden_layers': hidden_layers,
            'depth': len(hidden_layers),
            'width': max(hidden_layers) if hidden_layers else 0,
            'dataset': first_metadata.dataset_name,
            'sample_size': first_result.sample_size,
            'n_trials': len(trials),

            # Training accuracy (on train set)
            'train_accuracy_mean': float(np.mean(train_accs)) if train_accs else None,
            'train_accuracy_std': float(np.std(train_accs)) if train_accs else None,

            # Test accuracy (on held-out test set)
            'test_accuracy_mean': float(np.mean(test_accs)) if test_accs else None,
            'test_accuracy_std': float(np.std(test_accs)) if test_accs else None,

            # Generalization gap (train - test, positive = overfitting)
            'generalization_gap_mean': float(np.mean(gaps)) if gaps else None,
            'generalization_gap_std': float(np.std(gaps)) if gaps else None,

            # Overall final accuracy (same as train for backward compat)
            'mean_accuracy': float(np.mean(final_accs)),
            'std_accuracy': float(np.std(final_accs)),

            # Training time
            'mean_training_time': float(np.mean(training_times)),

            # Noise robustness
            'noise_drop': noise_drop,
            **noise_robustness_agg,
        })

    return aggregated


def export_phase7_csv(aggregated, filepath):
    """Export Phase 7 results to CSV with generalization columns."""
    # Define columns
    columns = [
        'config_key', 'element_name', 'activation', 'hidden_layers',
        'depth', 'width', 'dataset', 'sample_size', 'n_trials',
        'train_accuracy_mean', 'train_accuracy_std',
        'test_accuracy_mean', 'test_accuracy_std',
        'generalization_gap_mean', 'generalization_gap_std',
        'mean_accuracy', 'std_accuracy', 'mean_training_time',
        'noise_drop',
        'noise_0.0_mean', 'noise_0.0_std',
        'noise_0.1_mean', 'noise_0.1_std',
        'noise_0.2_mean', 'noise_0.2_std',
        'noise_0.3_mean', 'noise_0.3_std',
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in aggregated:
            # Convert hidden_layers to string
            row_copy = row.copy()
            row_copy['hidden_layers'] = str(row_copy['hidden_layers'])
            writer.writerow(row_copy)


def print_phase7_summary(aggregated, config):
    """Print Phase 7 specific summary with generalization insights."""
    print("=" * 70)
    print("PHASE 7 RESULTS SUMMARY: Generalization Study")
    print("=" * 70)

    print(f"\nTotal configurations: {len(aggregated)}")

    # Group by sample size
    print("\n   By Sample Size:")
    size_groups = defaultdict(list)
    for a in aggregated:
        size_groups[a['sample_size']].append(a)

    for size in sorted(size_groups.keys()):
        group = size_groups[size]
        avg_train = np.mean([a['train_accuracy_mean'] for a in group if a['train_accuracy_mean']]) * 100
        avg_test = np.mean([a['test_accuracy_mean'] for a in group if a['test_accuracy_mean']]) * 100
        avg_gap = np.mean([a['generalization_gap_mean'] for a in group if a['generalization_gap_mean']]) * 100
        avg_noise_drop = np.mean([a['noise_drop'] for a in group]) * 100

        print(f"      n={size:3d}: train={avg_train:.1f}%, test={avg_test:.1f}%, "
              f"gap={avg_gap:+.1f}%, noise_drop={avg_noise_drop:.1f}%")

    # Activation ranking by generalization
    print("\n   Activation Ranking (by Test Accuracy):")
    activation_groups = defaultdict(list)
    for a in aggregated:
        activation_groups[a['activation']].append(a)

    activation_summary = []
    for act, group in activation_groups.items():
        avg_train = np.mean([a['train_accuracy_mean'] for a in group if a['train_accuracy_mean']]) * 100
        avg_test = np.mean([a['test_accuracy_mean'] for a in group if a['test_accuracy_mean']]) * 100
        avg_gap = np.mean([a['generalization_gap_mean'] for a in group if a['generalization_gap_mean']]) * 100
        avg_noise_drop = np.mean([a['noise_drop'] for a in group]) * 100
        activation_summary.append({
            'activation': act,
            'train_acc': avg_train,
            'test_acc': avg_test,
            'gap': avg_gap,
            'noise_drop': avg_noise_drop,
        })

    for i, s in enumerate(sorted(activation_summary, key=lambda x: -x['test_acc']), 1):
        print(f"      {i}. {s['activation']:10s} train={s['train_acc']:.1f}%, "
              f"test={s['test_acc']:.1f}%, gap={s['gap']:+.1f}%, "
              f"noise_drop={s['noise_drop']:.1f}%")

    # Sample efficiency by activation
    print("\n   Sample Efficiency (accuracy at n=50 vs n=500):")
    for act in ['relu', 'sigmoid', 'sine', 'tanh']:
        small = [a for a in aggregated if a['activation'] == act and a['sample_size'] == 50]
        large = [a for a in aggregated if a['activation'] == act and a['sample_size'] == 500]

        if small and large:
            small_test = np.mean([a['test_accuracy_mean'] for a in small if a['test_accuracy_mean']]) * 100
            large_test = np.mean([a['test_accuracy_mean'] for a in large if a['test_accuracy_mean']]) * 100
            efficiency = small_test / large_test * 100 if large_test > 0 else 0

            print(f"      {act:10s}: n=50 {small_test:.1f}% -> n=500 {large_test:.1f}% "
                  f"(efficiency={efficiency:.0f}%)")

    # Depth effect on generalization
    print("\n   Depth Effect on Generalization Gap:")
    depth_groups = defaultdict(list)
    for a in aggregated:
        depth_groups[a['depth']].append(a)

    for depth in sorted(depth_groups.keys()):
        group = depth_groups[depth]
        avg_gap = np.mean([a['generalization_gap_mean'] for a in group if a['generalization_gap_mean']]) * 100
        avg_test = np.mean([a['test_accuracy_mean'] for a in group if a['test_accuracy_mean']]) * 100
        print(f"      depth={depth}: gap={avg_gap:+.1f}%, test={avg_test:.1f}%")

    # Dataset comparison
    print("\n   Dataset Comparison (Generalization Gap):")
    dataset_groups = defaultdict(list)
    for a in aggregated:
        dataset_groups[a['dataset']].append(a)

    for dataset, group in sorted(dataset_groups.items()):
        avg_gap = np.mean([a['generalization_gap_mean'] for a in group if a['generalization_gap_mean']]) * 100
        avg_test = np.mean([a['test_accuracy_mean'] for a in group if a['test_accuracy_mean']]) * 100
        print(f"      {dataset:10s}: gap={avg_gap:+.1f}%, test={avg_test:.1f}%")

    # Noise robustness ranking
    print("\n   Noise Robustness Ranking (accuracy drop at max noise):")
    for i, s in enumerate(sorted(activation_summary, key=lambda x: x['noise_drop']), 1):
        print(f"      {i}. {s['activation']:10s} noise_drop={s['noise_drop']:.1f}%")

    # Worst generalizers (highest gap)
    print("\n   Highest Generalization Gap (potential overfitting):")
    sorted_by_gap = sorted(
        [a for a in aggregated if a.get('generalization_gap_mean') is not None],
        key=lambda x: -x['generalization_gap_mean']
    )[:5]

    for a in sorted_by_gap:
        print(f"      {a['element_name']:20s} {a['dataset']:10s} n={a['sample_size']}: "
              f"gap={a['generalization_gap_mean']*100:+.1f}%")

    # Best sample efficiency
    print("\n   Best Sample Efficiency (high accuracy at n=50):")
    n50_configs = sorted(
        [a for a in aggregated if a['sample_size'] == 50 and a.get('test_accuracy_mean')],
        key=lambda x: -x['test_accuracy_mean']
    )[:5]

    for a in n50_configs:
        print(f"      {a['element_name']:20s} {a['dataset']:10s}: "
              f"test={a['test_accuracy_mean']*100:.1f}%")

    print("=" * 70)


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    output_dir = project_root / 'data'

    # Load config for reference
    config = Phase7Config()

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
        print("   Run examples/run_phase7_training.py first.")
        return

    print(f"\n   Found {completed} completed experiments")

    # Aggregate results
    print("\n   Aggregating results with generalization metrics...")
    aggregated = aggregate_phase7_experiments(store, status='completed')
    print(f"   Generated {len(aggregated)} configuration summaries")

    if not aggregated:
        print("\n   No Phase 7 experiments could be aggregated.")
        print("   (Experiments may be from earlier phases without generalization fields)")
        return

    # Export to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'phase7_summary_{timestamp}.csv'
    export_phase7_csv(aggregated, str(csv_path))
    print(f"\n   Exported Phase 7 results to: {csv_path}")

    # Also create a latest version
    latest_path = output_dir / 'phase7_summary.csv'
    export_phase7_csv(aggregated, str(latest_path))
    print(f"   Also saved as: {latest_path}")

    # Print summary
    print("\n")
    print_phase7_summary(aggregated, config)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review phase7_summary.csv for detailed generalization metrics")
    print("  2. Run examples/visualize_phase7.py for generalization charts")
    print("  3. Compare sample efficiency across activations")
    print("=" * 60)


if __name__ == '__main__':
    main()
