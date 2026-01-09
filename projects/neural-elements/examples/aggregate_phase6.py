#!/usr/bin/env python3
"""
Aggregate Phase 6 experiment results with dynamics metrics.

Run this after Phase 6 experiments complete to:
1. Aggregate results by configuration
2. Compute dynamics metrics (convergence, stability, gradient health)
3. Export to CSV for external analysis
4. Print summary with dynamics insights
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.persistence import ExperimentStore
from src.analysis.config_selection import Phase6Config
from src.analysis.dynamics import (
    compute_dynamics_metrics,
    aggregate_dynamics_across_trials,
    classify_training_pattern,
)

import numpy as np


def get_config_key(element_name, dataset, skip_connections=False):
    """Generate a unique key for grouping trials."""
    skip_suffix = "_skip" if skip_connections else ""
    return f"{element_name}_{dataset}{skip_suffix}"


def aggregate_phase6_experiments(store, status='completed'):
    """
    Aggregate Phase 6 experiments with dynamics metrics.

    Returns list of dicts with both accuracy stats and dynamics metrics.
    """
    # Get all completed experiments
    experiments = store.list_experiments(status=status, limit=10000)

    # Group by configuration
    config_groups = defaultdict(list)
    for exp in experiments:
        # Load full result to get history with gradient stats
        _, result = store.load_experiment(exp['experiment_id'])
        if result is None:
            continue

        # Load metadata for config details
        metadata = store.load_metadata(exp['experiment_id'])
        if metadata is None:
            continue

        skip_connections = metadata.element_config.get('skip_connections', False)
        config_key = get_config_key(
            exp['element_name'],
            exp.get('dataset', 'unknown'),
            skip_connections
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

        # Basic accuracy stats
        accuracies = [t['result'].final_accuracy for t in trials]
        losses = [t['result'].final_loss for t in trials]
        training_times = [t['result'].training_time_seconds for t in trials]

        # Get config details from first trial
        first_metadata = trials[0]['metadata']
        element_config = first_metadata.element_config
        hidden_layers = element_config.get('hidden_layers', [])
        skip_connections = element_config.get('skip_connections', False)

        # Compute dynamics metrics for each trial
        trial_dynamics = []
        for trial in trials:
            history = trial['result'].history
            if history:
                dynamics = compute_dynamics_metrics(history)
                trial_dynamics.append(dynamics)

        # Aggregate dynamics across trials
        dynamics_agg = aggregate_dynamics_across_trials(trial_dynamics) if trial_dynamics else {}

        # Classify architecture type
        arch_type = classify_architecture_type(hidden_layers, skip_connections)

        aggregated.append({
            'config_key': config_key,
            'element_name': trials[0]['result'].element_name,
            'activation': element_config.get('activation', 'unknown'),
            'hidden_layers': hidden_layers,
            'depth': len(hidden_layers),
            'width': max(hidden_layers) if hidden_layers else 0,
            'min_width': min(hidden_layers) if hidden_layers else 0,
            'skip_connections': skip_connections,
            'architecture_type': arch_type,
            'dataset': first_metadata.dataset_name,
            'n_trials': len(trials),

            # Accuracy stats
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),

            # Loss stats
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),

            # Training time
            'mean_training_time': float(np.mean(training_times)),

            # Dynamics metrics (aggregated across trials)
            **dynamics_agg,
        })

    return aggregated


def classify_architecture_type(hidden_layers, skip_connections):
    """Classify architecture into type categories."""
    if not hidden_layers:
        return 'linear'

    depth = len(hidden_layers)
    is_uniform = all(w == hidden_layers[0] for w in hidden_layers)

    if skip_connections:
        return 'skip'
    elif is_uniform:
        return 'uniform'
    elif depth == 3 and hidden_layers[1] < hidden_layers[0]:
        return 'bottleneck'
    else:
        return 'non-uniform'


def export_phase6_csv(aggregated, filepath):
    """Export Phase 6 results to CSV with dynamics columns."""
    import csv

    # Define columns
    columns = [
        'config_key', 'element_name', 'activation', 'hidden_layers',
        'depth', 'width', 'min_width', 'skip_connections', 'architecture_type',
        'dataset', 'n_trials',
        'mean_accuracy', 'std_accuracy', 'min_accuracy', 'max_accuracy',
        'mean_loss', 'std_loss', 'mean_training_time',
        # Dynamics columns
        'convergence_epoch_mean', 'convergence_epoch_std',
        'loss_stability_mean', 'loss_stability_std',
        'grad_mean_grad_overall_mean', 'grad_mean_grad_final_mean',
        'grad_grad_trend_slope_mean', 'grad_gradient_ratio_mean',
        'vanishing_gradient_rate',
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in aggregated:
            # Convert hidden_layers to string
            row_copy = row.copy()
            row_copy['hidden_layers'] = str(row_copy['hidden_layers'])
            writer.writerow(row_copy)


def print_phase6_summary(aggregated, config):
    """Print Phase 6 specific summary with dynamics insights."""
    print("=" * 70)
    print("PHASE 6 RESULTS SUMMARY: Learning Dynamics Study")
    print("=" * 70)

    print(f"\nTotal configurations: {len(aggregated)}")

    # Group by architecture type
    print("\n   By Architecture Type:")
    arch_groups = defaultdict(list)
    for a in aggregated:
        arch_groups[a['architecture_type']].append(a)

    for arch_type, group in sorted(arch_groups.items()):
        avg_acc = np.mean([a['mean_accuracy'] for a in group]) * 100
        avg_conv = np.mean([a.get('convergence_epoch_mean', 0) for a in group])
        vanishing_rate = np.mean([a.get('vanishing_gradient_rate', 0) for a in group]) * 100
        print(f"      {arch_type:12s}: {len(group):3d} configs, "
              f"avg acc {avg_acc:.1f}%, conv epoch {avg_conv:.0f}, "
              f"vanishing {vanishing_rate:.0f}%")

    # Activation ranking with dynamics
    print("\n   Activation Ranking (with Dynamics):")
    activation_groups = defaultdict(list)
    for a in aggregated:
        activation_groups[a['activation']].append(a)

    activation_summary = []
    for act, group in activation_groups.items():
        avg_acc = np.mean([a['mean_accuracy'] for a in group]) * 100
        avg_conv = np.mean([a.get('convergence_epoch_mean', 0) for a in group])
        avg_stability = np.mean([a.get('loss_stability_mean', 0) for a in group])
        vanishing_rate = np.mean([a.get('vanishing_gradient_rate', 0) for a in group]) * 100
        activation_summary.append({
            'activation': act,
            'accuracy': avg_acc,
            'convergence': avg_conv,
            'stability': avg_stability,
            'vanishing_rate': vanishing_rate,
        })

    for i, s in enumerate(sorted(activation_summary, key=lambda x: -x['accuracy']), 1):
        print(f"      {i}. {s['activation']:10s} acc={s['accuracy']:.1f}%, "
              f"conv={s['convergence']:.0f}, stab={s['stability']:.4f}, "
              f"vanish={s['vanishing_rate']:.0f}%")

    # Skip connection analysis (key Phase 4 question)
    print("\n   Skip Connection Effect (Phase 4 Key Question):")
    for activation in ['relu', 'sigmoid', 'sine', 'tanh']:
        baseline = [a for a in aggregated
                    if a['activation'] == activation
                    and a['architecture_type'] == 'uniform'
                    and a['depth'] in [5, 8]]
        skip = [a for a in aggregated
                if a['activation'] == activation
                and a['architecture_type'] == 'skip'
                and a['depth'] in [5, 8]]

        if baseline and skip:
            base_acc = np.mean([a['mean_accuracy'] for a in baseline]) * 100
            skip_acc = np.mean([a['mean_accuracy'] for a in skip]) * 100
            base_vanish = np.mean([a.get('vanishing_gradient_rate', 0) for a in baseline]) * 100
            skip_vanish = np.mean([a.get('vanishing_gradient_rate', 0) for a in skip]) * 100

            diff = skip_acc - base_acc
            diff_sign = "+" if diff > 0 else ""
            print(f"      {activation:10s}: base={base_acc:.1f}% -> skip={skip_acc:.1f}% "
                  f"({diff_sign}{diff:.1f}%) | "
                  f"vanish: {base_vanish:.0f}% -> {skip_vanish:.0f}%")

    # Bottleneck analysis
    print("\n   Bottleneck Dynamics (Phase 5 Key Question):")
    for arch_name in ['bottleneck_severe', 'bottleneck_extreme']:
        if arch_name == 'bottleneck_severe':
            target_layers = [32, 8, 32]
        else:
            target_layers = [8, 2, 8]

        matches = [a for a in aggregated
                   if a['hidden_layers'] == target_layers]

        if matches:
            avg_acc = np.mean([a['mean_accuracy'] for a in matches]) * 100
            avg_conv = np.mean([a.get('convergence_epoch_mean', 0) for a in matches])
            avg_grad_ratio = np.mean([a.get('grad_gradient_ratio_mean', 1) for a in matches])

            print(f"      {arch_name:20s} {str(target_layers):15s}: "
                  f"acc={avg_acc:.1f}%, conv={avg_conv:.0f}, "
                  f"grad_ratio={avg_grad_ratio:.2f}")

    # Convergence speed ranking
    print("\n   Fastest Converging Configurations:")
    sorted_by_conv = sorted(
        [a for a in aggregated if a.get('convergence_epoch_mean', float('inf')) < float('inf')],
        key=lambda x: x.get('convergence_epoch_mean', float('inf'))
    )[:5]

    for a in sorted_by_conv:
        print(f"      {a['element_name']:20s} {a['dataset']:10s}: "
              f"epoch {a.get('convergence_epoch_mean', 0):.0f} -> "
              f"{a['mean_accuracy']*100:.1f}%")

    # Most unstable configurations
    print("\n   Most Unstable Training:")
    sorted_by_stability = sorted(
        [a for a in aggregated if a.get('loss_stability_mean', 0) > 0],
        key=lambda x: -x.get('loss_stability_mean', 0)
    )[:5]

    for a in sorted_by_stability:
        print(f"      {a['element_name']:20s} {a['dataset']:10s}: "
              f"stability={a.get('loss_stability_mean', 0):.4f}, "
              f"acc={a['mean_accuracy']*100:.1f}%")

    print("=" * 70)


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    output_dir = project_root / 'data'

    # Load config for reference
    config = Phase6Config()

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
        print("   Run examples/run_phase6_training.py first.")
        return

    print(f"\n   Found {completed} completed experiments")

    # Aggregate results with dynamics
    print("\n   Aggregating results with dynamics metrics...")
    aggregated = aggregate_phase6_experiments(store, status='completed')
    print(f"   Generated {len(aggregated)} configuration summaries")

    if not aggregated:
        print("\n   No experiments could be aggregated.")
        return

    # Export to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'phase6_summary_{timestamp}.csv'
    export_phase6_csv(aggregated, str(csv_path))
    print(f"\n   Exported Phase 6 results to: {csv_path}")

    # Also create a latest version
    latest_path = output_dir / 'phase6_summary.csv'
    export_phase6_csv(aggregated, str(latest_path))
    print(f"   Also saved as: {latest_path}")

    # Print summary
    print("\n")
    print_phase6_summary(aggregated, config)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review phase6_summary.csv for detailed dynamics metrics")
    print("  2. Run examples/visualize_phase6.py for dynamics charts")
    print("  3. Compare gradient flow between activations")
    print("=" * 60)


if __name__ == '__main__':
    main()
