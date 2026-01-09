#!/usr/bin/env python3
"""
Phase 4: Extended Depth Study Training Run

This script runs the Phase 4 experiment matrix:
- 4 activations: sigmoid, relu, sine, tanh
- 4 depths: 6, 7, 8, 10 hidden layers
- Fixed width: 8 neurons per layer
- 4 datasets: xor, moons, circles, spirals
- 20 trials per configuration
- Two variants: baseline and with skip connections

Total: 4 × 4 × 4 × 20 × 2 = 2,560 experiments
Expected runtime: ~5-10 minutes on 8-core machine
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.jobs import JobManager
from src.core.persistence import ExperimentStore
from src.analysis.config_selection import Phase4Config, estimate_runtime


def run_variant(job_manager: JobManager, config: Phase4Config, variant_name: str) -> str:
    """Run a single variant (baseline or with skip connections)."""
    element_configs = config.get_element_configs()
    summary = config.get_summary()

    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}")
    print(f"{'='*60}")

    print(f"\n   Activations: {', '.join(summary['activations'])}")
    print(f"   Depths: {summary['depths']}")
    print(f"   Width: {summary['width']} neurons/layer")
    print(f"   Skip connections: {summary['skip_connections']}")
    print(f"   Datasets: {', '.join(summary['datasets'])}")
    print(f"   Trials per config: {summary['n_trials']}")
    print(f"   Total experiments: {summary['total_experiments']:,}")

    print("\n   Submitting job...")

    job_id = job_manager.submit_bulk_job(
        element_configs=element_configs,
        dataset_names=config.datasets,
        training_config=config.training_config,
        n_trials=config.n_trials,
    )

    print(f"   Job submitted: {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(description='Run Phase 4 training experiments')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials per configuration (default: 20)')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Only run baseline experiments (no skip connections)')
    parser.add_argument('--skip-only', action='store_true',
                        help='Only run skip connection experiments')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for completion instead of prompting')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'

    # Initialize store and job manager
    store = ExperimentStore(str(data_path))
    job_manager = JobManager(store, n_workers=None)  # Auto-detect workers

    # Calculate totals
    n_variants = 2
    if args.baseline_only or args.skip_only:
        n_variants = 1

    base_config = Phase4Config(n_trials=args.trials)
    experiments_per_variant = base_config.get_total_experiments()
    total_experiments = experiments_per_variant * n_variants

    runtime = estimate_runtime(
        n_experiments=total_experiments,
        n_workers=job_manager.n_workers,
        avg_seconds_per_experiment=0.4  # Deeper networks are slower
    )

    print("=" * 60)
    print("NEURAL ELEMENTS - Phase 4: Extended Depth Study")
    print("=" * 60)

    print("\n   Goal: Find depth limits and test skip connection rescue")

    print(f"\n   Scale:")
    print(f"   Variants to run: {n_variants}")
    print(f"   Experiments per variant: {experiments_per_variant:,}")
    print(f"   Total experiments: {total_experiments:,}")

    print(f"\n   Resources:")
    print(f"   Workers: {job_manager.n_workers}")
    print(f"   Est. time: {runtime['total_minutes']:.1f} minutes")

    print(f"\n   Training Config:")
    for key, value in base_config.training_config.items():
        print(f"   {key}: {value}")

    # Confirm
    print("\n" + "-" * 60)
    response = input("Start Phase 4 training? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    job_ids = []

    # Run baseline variant
    if not args.skip_only:
        baseline_config = Phase4Config(n_trials=args.trials, skip_connections=False)
        job_id = run_variant(job_manager, baseline_config, "Baseline (no skip connections)")
        job_ids.append(('baseline', job_id))

    # Run skip connections variant
    if not args.baseline_only:
        skip_config = Phase4Config(n_trials=args.trials, skip_connections=True)
        job_id = run_variant(job_manager, skip_config, "With Skip Connections")
        job_ids.append(('skip_connections', job_id))

    print("\n" + "=" * 60)
    print("Jobs submitted:")
    for variant_name, job_id in job_ids:
        print(f"   {variant_name}: {job_id}")

    print("\n   Monitor progress:")
    for variant_name, job_id in job_ids:
        print(f"   curl http://localhost:5000/api/bulk/jobs/{job_id}")

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
                status_lines = []

                for variant_name, job_id in job_ids:
                    job = job_manager.get_job_status(job_id)
                    if job is None:
                        status_lines.append(f"{variant_name}: NOT FOUND")
                        continue

                    progress = job.completed_runs + job.failed_runs
                    pct = 100 * progress / job.total_runs if job.total_runs > 0 else 0

                    status_icon = {
                        'running': '',
                        'completed': '',
                        'failed': '',
                        'cancelled': '',
                    }.get(job.status.value, '')

                    status_lines.append(
                        f"{variant_name}: {status_icon} {progress:,}/{job.total_runs:,} ({pct:.0f}%)"
                    )

                    if job.status.value not in ('completed', 'failed', 'cancelled'):
                        all_done = False

                elapsed = time.time() - start_time
                print(f"\r[{elapsed:.0f}s] " + " | ".join(status_lines) + "   ", end='', flush=True)

                if all_done:
                    print(f"\n\n   All jobs completed in {elapsed/60:.1f} minutes")
                    break

                time.sleep(2)

        except KeyboardInterrupt:
            print("\n\n   Stopped waiting. Jobs continue in background.")

    print("\n" + "=" * 60)
    print("Phase 4 jobs submitted. Next steps:")
    print("  1. Wait for experiments to complete")
    print("  2. Run aggregation: python examples/aggregate_phase4.py")
    print("  3. Generate visualizations: python examples/visualize_phase4.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
