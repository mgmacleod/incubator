#!/usr/bin/env python3
"""
Phase 5: Architecture Space Exploration Training Run

This script runs the Phase 5 experiment matrix:
- 11 non-uniform architectures (bottleneck, pyramid, parameter-matched)
- 4 activations: relu, sine, tanh, leaky_relu
- 4 datasets: xor, moons, circles, spirals
- 20 trials per configuration

Total: 11 × 4 × 4 × 20 = 3,520 experiments
Expected runtime: ~10-15 minutes on 8-core machine
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.jobs import JobManager
from src.core.persistence import ExperimentStore
from src.analysis.config_selection import Phase5Config, estimate_runtime


def main():
    parser = argparse.ArgumentParser(description='Run Phase 5 training experiments')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials per configuration (default: 20)')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for completion instead of prompting')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'

    # Initialize store and job manager
    store = ExperimentStore(str(data_path))
    job_manager = JobManager(store, n_workers=None)  # Auto-detect workers

    # Create config
    config = Phase5Config(n_trials=args.trials)
    summary = config.get_summary()
    total_experiments = summary['total_experiments']

    runtime = estimate_runtime(
        n_experiments=total_experiments,
        n_workers=job_manager.n_workers,
        avg_seconds_per_experiment=0.35  # Non-uniform architectures may vary
    )

    print("=" * 60)
    print("NEURAL ELEMENTS - Phase 5: Architecture Space Exploration")
    print("=" * 60)

    print("\n   Goal: Test non-uniform architectures (bottleneck, pyramid, etc.)")

    print(f"\n   Architectures ({len(summary['architectures'])}):")
    for name, layers in summary['architectures'].items():
        print(f"     {name}: {layers}")

    print(f"\n   Activations: {', '.join(summary['activations'])}")
    print(f"   Datasets: {', '.join(summary['datasets'])}")
    print(f"   Trials per config: {summary['n_trials']}")
    print(f"   Element configs: {summary['n_element_configs']}")
    print(f"   Total experiments: {total_experiments:,}")

    print(f"\n   Resources:")
    print(f"   Workers: {job_manager.n_workers}")
    print(f"   Est. time: {runtime['total_minutes']:.1f} minutes")

    print(f"\n   Training Config:")
    for key, value in config.training_config.items():
        print(f"   {key}: {value}")

    # Confirm
    print("\n" + "-" * 60)
    response = input("Start Phase 5 training? [y/N]: ").strip().lower()
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
    print("Phase 5 job submitted. Next steps:")
    print("  1. Wait for experiments to complete")
    print("  2. Run aggregation: python examples/aggregate_phase5.py")
    print("  3. Generate visualizations: python examples/visualize_phase5.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
