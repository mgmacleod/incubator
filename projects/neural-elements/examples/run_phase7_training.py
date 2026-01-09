#!/usr/bin/env python3
"""
Phase 7: Generalization Study Training Run

This script runs the Phase 7 experiment matrix to understand generalization:
- 4 activations: relu, sigmoid, sine, tanh
- 4 depths: 1, 3, 5, 8
- 4 datasets: xor, moons, circles, spirals
- 4 sample sizes: 50, 100, 200, 500
- 20 trials per configuration

New metrics recorded:
- Train/test accuracy split
- Generalization gap (train - test)
- Noise robustness at multiple levels

Total: ~5,120 experiments
Expected runtime: ~45-60 minutes on 16-core machine
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.jobs import JobManager
from src.core.persistence import ExperimentStore
from src.analysis.config_selection import Phase7Config, estimate_runtime


def main():
    parser = argparse.ArgumentParser(description='Run Phase 7 training experiments')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials per configuration (default: 20)')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for completion instead of prompting')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Run only a specific sample size (default: all)')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'

    # Initialize store and job manager
    store = ExperimentStore(str(data_path))
    job_manager = JobManager(store, n_workers=None)  # Auto-detect workers

    # Create config
    config = Phase7Config(n_trials=args.trials)

    # Override training config
    config.training_config = {
        'epochs': args.epochs,
        'learning_rate': 0.1,
        'record_every': 50,
    }

    # Handle single sample size mode
    if args.sample_size:
        config.sample_sizes = [args.sample_size]

    summary = config.get_summary()
    total_experiments = summary['total_experiments']

    # Phase 7 experiments are faster (smaller datasets on average)
    runtime = estimate_runtime(
        n_experiments=total_experiments,
        n_workers=job_manager.n_workers,
        avg_seconds_per_experiment=0.25  # Faster due to smaller datasets
    )

    print("=" * 60)
    print("NEURAL ELEMENTS - Phase 7: Generalization Study")
    print("=" * 60)

    print("\n   Goal: Test whether elements generalize or just memorize")
    print("   Recording: train/test accuracy, generalization gap, noise robustness")

    print(f"\n   Activations: {', '.join(summary['activations'])}")
    print(f"   Depths: {summary['depths']}")
    print(f"   Width: {summary['width']}")

    print(f"\n   Generalization parameters:")
    print(f"     Train/test split: {summary['train_split']:.0%} / {1-summary['train_split']:.0%}")
    print(f"     Sample sizes: {summary['sample_sizes']}")
    print(f"     Noise levels: {summary['noise_levels']}")

    print(f"\n   Datasets: {', '.join(summary['datasets'])}")
    print(f"   Trials per config: {summary['n_trials']}")
    print(f"   Element configs: {summary['n_element_configs']}")
    print(f"   Total experiments: {total_experiments:,}")

    print(f"\n   Resources:")
    print(f"   Workers: {job_manager.n_workers}")
    print(f"   Est. time: {runtime['total_minutes']:.1f} minutes")

    print(f"\n   Training Config:")
    for key, value in config.training_config.items():
        print(f"     {key}: {value}")

    # Confirm
    print("\n" + "-" * 60)
    response = input("Start Phase 7 training? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    # Get element configs
    element_configs = config.get_element_configs()

    # Submit one job per sample size
    job_ids = []
    for sample_size in config.sample_sizes:
        phase7_config = {
            'train_split': config.train_split,
            'noise_levels': config.noise_levels,
            'sample_size': sample_size,
        }

        print(f"\n   Submitting job for sample_size={sample_size}...")

        job_id = job_manager.submit_phase7_job(
            element_configs=element_configs,
            dataset_names=config.datasets,
            training_config=config.training_config,
            phase7_config=phase7_config,
            n_trials=config.n_trials,
        )

        job_ids.append((sample_size, job_id))
        print(f"   Job submitted: {job_id}")

    print(f"\n   Total jobs submitted: {len(job_ids)}")
    print("\n   Monitor progress:")
    for sample_size, job_id in job_ids:
        print(f"   [size={sample_size}] curl http://localhost:5000/api/bulk/jobs/{job_id}")

    # Option to wait
    print("\n" + "-" * 60)
    if args.wait:
        wait = 'y'
    else:
        wait = input("Wait for completion? [y/N]: ").strip().lower()

    if wait == 'y':
        print("\n   Waiting for jobs to complete...")
        print("   (Press Ctrl+C to stop waiting - jobs continue in background)\n")

        start_time = time.time()

        try:
            while True:
                all_done = True
                total_completed = 0
                total_runs = 0

                for sample_size, job_id in job_ids:
                    job = job_manager.get_job_status(job_id)
                    if job is None:
                        continue

                    total_completed += job.completed_runs + job.failed_runs
                    total_runs += job.total_runs

                    if job.status.value not in ('completed', 'failed', 'cancelled'):
                        all_done = False

                pct = 100 * total_completed / total_runs if total_runs > 0 else 0
                elapsed = time.time() - start_time
                print(f"\r[{elapsed:.0f}s] {total_completed:,}/{total_runs:,} ({pct:.0f}%)   ", end='', flush=True)

                if all_done:
                    print(f"\n\n   All jobs completed in {elapsed/60:.1f} minutes")
                    break

                time.sleep(2)

        except KeyboardInterrupt:
            print("\n\n   Stopped waiting. Jobs continue in background.")

    print("\n" + "=" * 60)
    print("Phase 7 jobs submitted. Next steps:")
    print("  1. Wait for experiments to complete")
    print("  2. Run aggregation: python examples/aggregate_phase7.py")
    print("  3. Generate visualizations: python examples/visualize_phase7.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
