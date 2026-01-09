#!/usr/bin/env python3
"""
Phase 8: Combination & Transfer Training Run

This script runs the Phase 8 experiment matrix to test composability:
- Stacking: Train element A, freeze, add element B on top
- Transfer: Train on dataset X, fine-tune on dataset Y
- Ensembles: Combine predictions from multiple elements

Total: ~2,000 experiments
Expected runtime: ~40 minutes on 16-core machine
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.persistence import ExperimentStore, ExperimentMetadata
from src.core.jobs import (
    phase8_stacking_worker,
    phase8_transfer_worker,
    phase8_ensemble_worker,
)
from src.analysis.config_selection import Phase8Config, estimate_runtime
from src.datasets.toy import xor_dataset, spirals, moons, circles


def get_dataset(name: str, n_samples: int = 500):
    """Load a dataset by name."""
    if name == 'xor':
        return xor_dataset(n_samples=n_samples)
    elif name == 'spirals':
        return spirals(n_samples=n_samples)
    elif name == 'moons':
        return moons(n_samples=n_samples)
    elif name == 'circles':
        return circles(n_samples=n_samples)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def run_stacking_experiments(config: Phase8Config, store: ExperimentStore,
                             n_workers: int, n_trials: int):
    """Run stacking experiments."""
    print("\n" + "=" * 60)
    print("STACKING EXPERIMENTS")
    print("=" * 60)

    stacking_configs = config.get_stacking_configs()
    print(f"Configurations: {len(stacking_configs)}")
    print(f"Trials per config: {n_trials}")
    print(f"Total experiments: {len(stacking_configs) * n_trials}")

    # Prepare tasks
    tasks = []
    for stacking_config in stacking_configs:
        dataset_name = stacking_config['dataset']
        X, y = get_dataset(dataset_name)

        for trial in range(n_trials):
            experiment_id = f"stack_{stacking_config['activation']}_{stacking_config['bottom_depth']}_{stacking_config['top_depth']}_{dataset_name}_{trial}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

            # Save metadata
            metadata = ExperimentMetadata(
                experiment_id=experiment_id,
                job_id=None,
                created_at=datetime.now().isoformat(),
                element_name=f"stack-{stacking_config['activation'][:3]}-{stacking_config['bottom_depth']}+{stacking_config['top_depth']}",
                element_config=stacking_config,
                dataset_name=dataset_name,
                training_config=config.training_config,
                status='pending',
            )
            store.save_metadata(metadata)

            tasks.append((
                stacking_config,
                config.training_config,
                experiment_id,
                str(store.base_path),
                X,
                y,
            ))

    # Run experiments
    print(f"\nRunning {len(tasks)} stacking experiments with {n_workers} workers...")
    start_time = time.time()

    with Pool(n_workers) as pool:
        results = pool.map(phase8_stacking_worker, tasks)

    elapsed = time.time() - start_time
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] == 'failed')

    print(f"\nCompleted: {completed}, Failed: {failed}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per experiment)")

    return results


def run_transfer_experiments(config: Phase8Config, store: ExperimentStore,
                             n_workers: int, n_trials: int):
    """Run transfer learning experiments."""
    print("\n" + "=" * 60)
    print("TRANSFER EXPERIMENTS")
    print("=" * 60)

    transfer_configs = config.get_transfer_configs()
    print(f"Configurations: {len(transfer_configs)}")
    print(f"Trials per config: {n_trials}")
    print(f"Total experiments: {len(transfer_configs) * n_trials}")

    # Pre-load all datasets
    datasets_dict = {}
    for dataset_name in config.datasets:
        X, y = get_dataset(dataset_name)
        datasets_dict[dataset_name] = (X, y)

    # Prepare tasks
    tasks = []
    for transfer_config in transfer_configs:
        for trial in range(n_trials):
            experiment_id = f"transfer_{transfer_config['activation']}_{transfer_config['source_dataset']}_{transfer_config['target_dataset']}_{transfer_config['freeze_mode']}_{trial}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

            # Save metadata
            metadata = ExperimentMetadata(
                experiment_id=experiment_id,
                job_id=None,
                created_at=datetime.now().isoformat(),
                element_name=f"transfer-{transfer_config['activation'][:3]}-{transfer_config['source_dataset'][:3]}to{transfer_config['target_dataset'][:3]}",
                element_config=transfer_config,
                dataset_name=transfer_config['target_dataset'],
                training_config=config.training_config,
                status='pending',
            )
            store.save_metadata(metadata)

            tasks.append((
                transfer_config,
                config.training_config,
                experiment_id,
                str(store.base_path),
                datasets_dict,
            ))

    # Run experiments
    print(f"\nRunning {len(tasks)} transfer experiments with {n_workers} workers...")
    start_time = time.time()

    with Pool(n_workers) as pool:
        results = pool.map(phase8_transfer_worker, tasks)

    elapsed = time.time() - start_time
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] == 'failed')

    print(f"\nCompleted: {completed}, Failed: {failed}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per experiment)")

    return results


def run_ensemble_experiments(config: Phase8Config, store: ExperimentStore,
                             n_workers: int, n_trials: int):
    """Run ensemble experiments."""
    print("\n" + "=" * 60)
    print("ENSEMBLE EXPERIMENTS")
    print("=" * 60)

    ensemble_configs = config.get_ensemble_configs()
    print(f"Configurations: {len(ensemble_configs)}")
    print(f"Trials per config: {n_trials}")
    print(f"Total experiments: {len(ensemble_configs) * n_trials}")

    # Prepare tasks
    tasks = []
    for ensemble_config in ensemble_configs:
        dataset_name = ensemble_config['dataset']
        X, y = get_dataset(dataset_name)

        for trial in range(n_trials):
            experiment_id = f"ensemble_{ensemble_config['activation']}_{ensemble_config['ensemble_type']}_{ensemble_config['ensemble_size']}_{dataset_name}_{trial}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

            # Save metadata
            metadata = ExperimentMetadata(
                experiment_id=experiment_id,
                job_id=None,
                created_at=datetime.now().isoformat(),
                element_name=f"ensemble-{ensemble_config['activation'][:3]}-{ensemble_config['ensemble_type']}-{ensemble_config['ensemble_size']}",
                element_config=ensemble_config,
                dataset_name=dataset_name,
                training_config=config.training_config,
                status='pending',
            )
            store.save_metadata(metadata)

            tasks.append((
                ensemble_config,
                config.training_config,
                experiment_id,
                str(store.base_path),
                X,
                y,
            ))

    # Run experiments
    print(f"\nRunning {len(tasks)} ensemble experiments with {n_workers} workers...")
    start_time = time.time()

    with Pool(n_workers) as pool:
        results = pool.map(phase8_ensemble_worker, tasks)

    elapsed = time.time() - start_time
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] == 'failed')

    print(f"\nCompleted: {completed}, Failed: {failed}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per experiment)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run Phase 8 training experiments')
    parser.add_argument('--type', type=str, default='all',
                        choices=['stacking', 'transfer', 'ensemble', 'all'],
                        help='Type of experiment to run (default: all)')
    parser.add_argument('--trials', type=int, default=None,
                        help='Number of trials per configuration (overrides config)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'

    # Initialize store
    store = ExperimentStore(str(data_path))

    # Create config
    config = Phase8Config()

    # Override training config
    config.training_config = {
        'epochs': args.epochs,
        'learning_rate': 0.1,
        'record_every': 50,
    }

    # Determine number of workers
    n_workers = args.workers or max(1, cpu_count() - 1)

    # Determine trials
    stacking_trials = args.trials or config.n_trials
    transfer_trials = args.trials or config.n_trials
    ensemble_trials = args.trials or config.ensemble_trials

    summary = config.get_summary()
    totals = config.get_total_experiments()

    print("=" * 60)
    print("NEURAL ELEMENTS - Phase 8: Combination & Transfer")
    print("=" * 60)

    print("\n   Goal: Test whether element properties are composable")
    print("\n   Experiment types:")
    print(f"     Stacking: {summary['stacking']['experiments']} experiments")
    print(f"       - Bottom depths: {summary['stacking']['bottom_depths']}")
    print(f"       - Top depths: {summary['stacking']['top_depths']}")
    print(f"     Transfer: {summary['transfer']['experiments']} experiments")
    print(f"       - Pairs: {summary['transfer']['pairs']}")
    print(f"       - Freeze modes: {summary['transfer']['freeze_modes']}")
    print(f"     Ensemble: {summary['ensemble']['experiments']} experiments")
    print(f"       - Sizes: {summary['ensemble']['sizes']}")
    print(f"       - Types: {summary['ensemble']['types']}")

    print(f"\n   Common parameters:")
    print(f"     Activations: {', '.join(summary['activations'])}")
    print(f"     Datasets: {', '.join(summary['datasets'])}")
    print(f"     Width: {summary['width']}")

    print(f"\n   Resources:")
    print(f"     Workers: {n_workers}")

    print(f"\n   Training Config:")
    for key, value in config.training_config.items():
        print(f"     {key}: {value}")

    # Confirm
    print("\n" + "-" * 60)
    if args.type == 'all':
        response = input(f"Run ALL Phase 8 experiments (~{totals['total']} total)? [y/N]: ").strip().lower()
    else:
        response = input(f"Run {args.type} experiments? [y/N]: ").strip().lower()

    if response != 'y':
        print("Aborted.")
        return

    # Run experiments
    all_results = []

    if args.type in ('stacking', 'all'):
        results = run_stacking_experiments(config, store, n_workers, stacking_trials)
        all_results.extend(results)

    if args.type in ('transfer', 'all'):
        results = run_transfer_experiments(config, store, n_workers, transfer_trials)
        all_results.extend(results)

    if args.type in ('ensemble', 'all'):
        results = run_ensemble_experiments(config, store, n_workers, ensemble_trials)
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 8 COMPLETE")
    print("=" * 60)

    total_completed = sum(1 for r in all_results if r['status'] == 'completed')
    total_failed = sum(1 for r in all_results if r['status'] == 'failed')

    print(f"\n   Total experiments: {len(all_results)}")
    print(f"   Completed: {total_completed}")
    print(f"   Failed: {total_failed}")

    print("\n   Next steps:")
    print("     1. Run aggregation: python examples/aggregate_phase8.py")
    print("     2. Generate visualizations: python examples/visualize_phase8.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
