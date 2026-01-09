#!/usr/bin/env python3
"""
Aggregate Phase 8 experiment results with combination/transfer metrics.

Run this after Phase 8 experiments complete to:
1. Aggregate results by experiment type (stacking, transfer, ensemble)
2. Compute composability metrics
3. Export to CSV for external analysis
4. Print summary with key insights
"""

import sys
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.persistence import ExperimentStore
from src.analysis.config_selection import Phase8Config

import numpy as np


def aggregate_stacking_experiments(store, status='completed'):
    """Aggregate stacking experiments by configuration."""
    experiments = store.list_experiments(status=status, limit=20000)

    # Group by configuration
    config_groups = defaultdict(list)
    for exp in experiments:
        _, result = store.load_experiment(exp['experiment_id'])
        if result is None or result.experiment_type != 'stacking':
            continue

        # Group key: activation, bottom_depth, top_depth, dataset
        metadata = store.load_metadata(exp['experiment_id'])
        if metadata is None:
            continue

        element_config = metadata.element_config
        key = (
            element_config.get('activation', 'unknown'),
            element_config.get('bottom_depth', 0),
            element_config.get('top_depth', 0),
            metadata.dataset_name,
        )

        config_groups[key].append({
            'result': result,
            'metadata': metadata,
        })

    # Aggregate each configuration
    aggregated = []
    for (activation, bottom_depth, top_depth, dataset), trials in config_groups.items():
        if not trials:
            continue

        bottom_accs = [t['result'].stack_bottom_accuracy for t in trials if t['result'].stack_bottom_accuracy is not None]
        combined_accs = [t['result'].stack_combined_accuracy for t in trials if t['result'].stack_combined_accuracy is not None]
        improvements = [t['result'].stack_improvement for t in trials if t['result'].stack_improvement is not None]

        aggregated.append({
            'experiment_type': 'stacking',
            'activation': activation,
            'bottom_depth': bottom_depth,
            'top_depth': top_depth,
            'dataset': dataset,
            'n_trials': len(trials),
            'bottom_accuracy_mean': float(np.mean(bottom_accs)) if bottom_accs else None,
            'bottom_accuracy_std': float(np.std(bottom_accs)) if bottom_accs else None,
            'combined_accuracy_mean': float(np.mean(combined_accs)) if combined_accs else None,
            'combined_accuracy_std': float(np.std(combined_accs)) if combined_accs else None,
            'improvement_mean': float(np.mean(improvements)) if improvements else None,
            'improvement_std': float(np.std(improvements)) if improvements else None,
        })

    return aggregated


def aggregate_transfer_experiments(store, status='completed'):
    """Aggregate transfer experiments by configuration."""
    experiments = store.list_experiments(status=status, limit=20000)

    # Group by configuration
    config_groups = defaultdict(list)
    for exp in experiments:
        _, result = store.load_experiment(exp['experiment_id'])
        if result is None or result.experiment_type != 'transfer':
            continue

        metadata = store.load_metadata(exp['experiment_id'])
        if metadata is None:
            continue

        element_config = metadata.element_config
        key = (
            element_config.get('activation', 'unknown'),
            result.source_dataset,
            result.target_dataset,
            result.freeze_mode,
        )

        config_groups[key].append({
            'result': result,
            'metadata': metadata,
        })

    # Aggregate each configuration
    aggregated = []
    for (activation, source, target, freeze_mode), trials in config_groups.items():
        if not trials:
            continue

        pretrained_accs = [t['result'].pretrained_accuracy for t in trials if t['result'].pretrained_accuracy is not None]
        scratch_accs = [t['result'].scratch_accuracy for t in trials if t['result'].scratch_accuracy is not None]
        benefits = [t['result'].transfer_benefit for t in trials if t['result'].transfer_benefit is not None]

        aggregated.append({
            'experiment_type': 'transfer',
            'activation': activation,
            'source_dataset': source,
            'target_dataset': target,
            'freeze_mode': freeze_mode,
            'n_trials': len(trials),
            'pretrained_accuracy_mean': float(np.mean(pretrained_accs)) if pretrained_accs else None,
            'pretrained_accuracy_std': float(np.std(pretrained_accs)) if pretrained_accs else None,
            'scratch_accuracy_mean': float(np.mean(scratch_accs)) if scratch_accs else None,
            'scratch_accuracy_std': float(np.std(scratch_accs)) if scratch_accs else None,
            'transfer_benefit_mean': float(np.mean(benefits)) if benefits else None,
            'transfer_benefit_std': float(np.std(benefits)) if benefits else None,
        })

    return aggregated


def aggregate_ensemble_experiments(store, status='completed'):
    """Aggregate ensemble experiments by configuration."""
    experiments = store.list_experiments(status=status, limit=20000)

    # Group by configuration
    config_groups = defaultdict(list)
    for exp in experiments:
        _, result = store.load_experiment(exp['experiment_id'])
        if result is None or result.experiment_type != 'ensemble':
            continue

        metadata = store.load_metadata(exp['experiment_id'])
        if metadata is None:
            continue

        element_config = metadata.element_config
        key = (
            element_config.get('activation', 'unknown'),
            result.ensemble_type,
            result.ensemble_size,
            metadata.dataset_name,
        )

        config_groups[key].append({
            'result': result,
            'metadata': metadata,
        })

    # Aggregate each configuration
    aggregated = []
    for (activation, ensemble_type, ensemble_size, dataset), trials in config_groups.items():
        if not trials:
            continue

        ensemble_accs = [t['result'].ensemble_accuracy for t in trials if t['result'].ensemble_accuracy is not None]
        improvements = [t['result'].ensemble_improvement for t in trials if t['result'].ensemble_improvement is not None]

        # Get individual accuracies (flatten all trials)
        all_individual_accs = []
        for t in trials:
            if t['result'].individual_accuracies:
                all_individual_accs.extend(t['result'].individual_accuracies)

        aggregated.append({
            'experiment_type': 'ensemble',
            'activation': activation,
            'ensemble_type': ensemble_type,
            'ensemble_size': ensemble_size,
            'dataset': dataset,
            'n_trials': len(trials),
            'ensemble_accuracy_mean': float(np.mean(ensemble_accs)) if ensemble_accs else None,
            'ensemble_accuracy_std': float(np.std(ensemble_accs)) if ensemble_accs else None,
            'individual_accuracy_mean': float(np.mean(all_individual_accs)) if all_individual_accs else None,
            'improvement_mean': float(np.mean(improvements)) if improvements else None,
            'improvement_std': float(np.std(improvements)) if improvements else None,
        })

    return aggregated


def export_phase8_csv(aggregated, filepath, exp_type):
    """Export Phase 8 results to CSV."""
    if not aggregated:
        return

    # Get all keys from first item
    fieldnames = list(aggregated[0].keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated:
            writer.writerow(row)


def print_phase8_summary(stacking, transfer, ensemble):
    """Print Phase 8 summary with key insights."""
    print("=" * 70)
    print("PHASE 8 RESULTS SUMMARY: Combination & Transfer")
    print("=" * 70)

    # Stacking summary
    if stacking:
        print("\n--- STACKING EXPERIMENTS ---")
        print(f"Total configurations: {len(stacking)}")

        # Group by activation
        by_activation = defaultdict(list)
        for s in stacking:
            by_activation[s['activation']].append(s)

        print("\nBy Activation (average improvement):")
        for act in sorted(by_activation.keys()):
            improvements = [s['improvement_mean'] for s in by_activation[act] if s['improvement_mean'] is not None]
            if improvements:
                avg_imp = np.mean(improvements) * 100
                print(f"   {act:10s}: {avg_imp:+.1f}%")

        # Best stacking combinations
        print("\nBest Stacking Combinations:")
        sorted_by_combined = sorted(
            [s for s in stacking if s['combined_accuracy_mean'] is not None],
            key=lambda x: -x['combined_accuracy_mean']
        )[:5]
        for s in sorted_by_combined:
            print(f"   {s['activation']}-{s['bottom_depth']}+{s['top_depth']} on {s['dataset']}: "
                  f"{s['combined_accuracy_mean']*100:.1f}% (improvement: {s['improvement_mean']*100:+.1f}%)")

    # Transfer summary
    if transfer:
        print("\n--- TRANSFER EXPERIMENTS ---")
        print(f"Total configurations: {len(transfer)}")

        # Group by source->target pair
        by_pair = defaultdict(list)
        for t in transfer:
            pair = f"{t['source_dataset']}->{t['target_dataset']}"
            by_pair[pair].append(t)

        print("\nBy Transfer Pair (average benefit):")
        for pair in sorted(by_pair.keys()):
            benefits = [t['transfer_benefit_mean'] for t in by_pair[pair] if t['transfer_benefit_mean'] is not None]
            if benefits:
                avg_benefit = np.mean(benefits) * 100
                print(f"   {pair:20s}: {avg_benefit:+.1f}%")

        # Best transfers
        print("\nBest Transfer Results:")
        sorted_by_benefit = sorted(
            [t for t in transfer if t['transfer_benefit_mean'] is not None],
            key=lambda x: -x['transfer_benefit_mean']
        )[:5]
        for t in sorted_by_benefit:
            print(f"   {t['activation']} {t['source_dataset']}->{t['target_dataset']} ({t['freeze_mode']}): "
                  f"pretrained={t['pretrained_accuracy_mean']*100:.1f}%, "
                  f"scratch={t['scratch_accuracy_mean']*100:.1f}%, "
                  f"benefit={t['transfer_benefit_mean']*100:+.1f}%")

        # Worst transfers (negative benefit)
        print("\nWorst Transfer Results (pretrained hurt):")
        sorted_by_benefit_asc = sorted(
            [t for t in transfer if t['transfer_benefit_mean'] is not None],
            key=lambda x: x['transfer_benefit_mean']
        )[:5]
        for t in sorted_by_benefit_asc:
            if t['transfer_benefit_mean'] < 0:
                print(f"   {t['activation']} {t['source_dataset']}->{t['target_dataset']} ({t['freeze_mode']}): "
                      f"benefit={t['transfer_benefit_mean']*100:+.1f}%")

    # Ensemble summary
    if ensemble:
        print("\n--- ENSEMBLE EXPERIMENTS ---")
        print(f"Total configurations: {len(ensemble)}")

        # Group by ensemble type
        by_type = defaultdict(list)
        for e in ensemble:
            by_type[e['ensemble_type']].append(e)

        print("\nBy Ensemble Type (average improvement):")
        for etype in sorted(by_type.keys()):
            improvements = [e['improvement_mean'] for e in by_type[etype] if e['improvement_mean'] is not None]
            if improvements:
                avg_imp = np.mean(improvements) * 100
                print(f"   {etype:10s}: {avg_imp:+.1f}%")

        # By ensemble size
        by_size = defaultdict(list)
        for e in ensemble:
            by_size[e['ensemble_size']].append(e)

        print("\nBy Ensemble Size (average improvement):")
        for size in sorted(by_size.keys()):
            improvements = [e['improvement_mean'] for e in by_size[size] if e['improvement_mean'] is not None]
            if improvements:
                avg_imp = np.mean(improvements) * 100
                print(f"   size={size}: {avg_imp:+.1f}%")

        # Best ensembles
        print("\nBest Ensemble Results:")
        sorted_by_acc = sorted(
            [e for e in ensemble if e['ensemble_accuracy_mean'] is not None],
            key=lambda x: -x['ensemble_accuracy_mean']
        )[:5]
        for e in sorted_by_acc:
            print(f"   {e['activation']} {e['ensemble_type']}-{e['ensemble_size']} on {e['dataset']}: "
                  f"{e['ensemble_accuracy_mean']*100:.1f}% (improvement: {e['improvement_mean']*100:+.1f}%)")

    print("\n" + "=" * 70)


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

    completed = stats['by_status'].get('completed', 0)
    if completed == 0:
        print("\n   No completed experiments found.")
        print("   Run examples/run_phase8_training.py first.")
        return

    # Aggregate by type
    print("\n   Aggregating Phase 8 results...")

    stacking = aggregate_stacking_experiments(store)
    print(f"   Stacking: {len(stacking)} configurations")

    transfer = aggregate_transfer_experiments(store)
    print(f"   Transfer: {len(transfer)} configurations")

    ensemble = aggregate_ensemble_experiments(store)
    print(f"   Ensemble: {len(ensemble)} configurations")

    total_configs = len(stacking) + len(transfer) + len(ensemble)
    if total_configs == 0:
        print("\n   No Phase 8 experiments found.")
        print("   (Experiments may be from earlier phases)")
        return

    # Export to CSVs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if stacking:
        csv_path = output_dir / f'phase8_stacking_{timestamp}.csv'
        export_phase8_csv(stacking, str(csv_path), 'stacking')
        print(f"\n   Exported stacking results to: {csv_path}")

        latest_path = output_dir / 'phase8_stacking.csv'
        export_phase8_csv(stacking, str(latest_path), 'stacking')

    if transfer:
        csv_path = output_dir / f'phase8_transfer_{timestamp}.csv'
        export_phase8_csv(transfer, str(csv_path), 'transfer')
        print(f"   Exported transfer results to: {csv_path}")

        latest_path = output_dir / 'phase8_transfer.csv'
        export_phase8_csv(transfer, str(latest_path), 'transfer')

    if ensemble:
        csv_path = output_dir / f'phase8_ensemble_{timestamp}.csv'
        export_phase8_csv(ensemble, str(csv_path), 'ensemble')
        print(f"   Exported ensemble results to: {csv_path}")

        latest_path = output_dir / 'phase8_ensemble.csv'
        export_phase8_csv(ensemble, str(latest_path), 'ensemble')

    # Print summary
    print("\n")
    print_phase8_summary(stacking, transfer, ensemble)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review phase8_*.csv files for detailed metrics")
    print("  2. Run examples/visualize_phase8.py for charts")
    print("  3. Compare composability across activations")
    print("=" * 60)


if __name__ == '__main__':
    main()
