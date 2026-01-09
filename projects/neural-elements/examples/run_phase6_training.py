#!/usr/bin/env python3
"""
Phase 6: Learning Dynamics Study Training Run

This script runs the Phase 6 experiment matrix to understand HOW elements learn:
- 4 activations: relu, sigmoid, sine, tanh
- 4 depths: 1, 3, 5, 8 (baseline)
- Skip connection variants at depths 5, 8
- Bottleneck architectures: [32,8,32], [8,2,8]
- 4 datasets: xor, moons, circles, spirals
- 20 trials per configuration

Total: ~2,560 experiments
Expected runtime: ~30-40 minutes on 8-core machine (longer due to gradient recording)
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.jobs import JobManager
from src.core.persistence import ExperimentStore
from src.analysis.config_selection import Phase6Config, estimate_runtime


def main():
    parser = argparse.ArgumentParser(description='Run Phase 6 training experiments')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials per configuration (default: 20)')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for completion instead of prompting')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'

    # Initialize store and job manager
    store = ExperimentStore(str(data_path))
    job_manager = JobManager(store, n_workers=None)  # Auto-detect workers

    # Create config with custom values
    training_config = {
        'epochs': args.epochs,
        'learning_rate': 0.1,
        'record_every': 10,
        'record_gradients': True,
        'record_weight_stats': True,
    }

    config = Phase6Config(n_trials=args.trials)
    # Override training config
    config.training_config = training_config

    summary = config.get_summary()
    total_experiments = summary['total_experiments']

    # Phase 6 experiments take longer due to gradient recording
    runtime = estimate_runtime(
        n_experiments=total_experiments,
        n_workers=job_manager.n_workers,
        avg_seconds_per_experiment=0.6  # ~2x slower due to extra recording
    )

    print("=" * 60)
    print("NEURAL ELEMENTS - Phase 6: Learning Dynamics Study")
    print("=" * 60)

    print("\n   Goal: Understand HOW elements learn, not just final accuracy")
    print("   Recording: gradient norms, weight evolution, convergence speed")

    print(f"\n   Activations: {', '.join(summary['activations'])}")
    print(f"   Depths: {summary['depths']}")
    print(f"   Width: {summary['width']}")

    print(f"\n   Configuration breakdown:")
    print(f"     Baseline (uniform): {summary['breakdown']['baseline']} configs")
    print(f"     Skip variants: {summary['breakdown']['skip_variants']} configs")
    print(f"     Bottleneck: {summary['breakdown']['bottleneck']} configs")

    print(f"\n   Bottleneck architectures:")
    for name, layers in summary['bottleneck_architectures'].items():
        print(f"     {name}: {layers}")

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
    response = input("Start Phase 6 training? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    # Get element configs
    element_configs = config.get_element_configs()

    print(f"\n   Submitting job with {len(element_configs)} element configs...")

    job_id = job_manager.submit_bulk_job(
        element_configs=element_configs,
        dataset_names=config.datasets,
        training_config=config.training_config,
        n_trials=config.n_trials,
    )

    print(f"   Job submitted: {job_id}")

    print("\n   Monitor progress:")
    print(f"   curl http://localhost:5000/api/bulk/jobs/{job_id}")

    # Option to wait
    print("\n" + "-" * 60)
    if args.wait:
        wait = 'y'
    else:
        wait = input("Wait for completion? [y/N]: ").strip().lower()

    if wait == 'y':
        print("\n   Waiting for job to complete...")
        print("   (Press Ctrl+C to stop waiting - job continues in background)\n")

        start_time = time.time()

        try:
            while True:
                job = job_manager.get_job_status(job_id)
                if job is None:
                    print("\n   Job not found!")
                    break

                progress = job.completed_runs + job.failed_runs
                pct = 100 * progress / job.total_runs if job.total_runs > 0 else 0

                status_icon = {
                    'running': '',
                    'completed': '',
                    'failed': '',
                    'cancelled': '',
                }.get(job.status.value, '')

                elapsed = time.time() - start_time
                print(f"\r[{elapsed:.0f}s] {status_icon} {progress:,}/{job.total_runs:,} ({pct:.0f}%)   ", end='', flush=True)

                if job.status.value in ('completed', 'failed', 'cancelled'):
                    print(f"\n\n   Job {job.status.value} in {elapsed/60:.1f} minutes")
                    break

                time.sleep(2)

        except KeyboardInterrupt:
            print("\n\n   Stopped waiting. Job continues in background.")

    print("\n" + "=" * 60)
    print("Phase 6 job submitted. Next steps:")
    print("  1. Wait for experiments to complete")
    print("  2. Run aggregation: python examples/aggregate_phase6.py")
    print("  3. Generate visualizations: python examples/visualize_phase6.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
