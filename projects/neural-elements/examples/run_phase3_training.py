#!/usr/bin/env python3
"""
Phase 3: Statistical Robustness Training Run

This script runs the Phase 3 experiment matrix:
- 6 activations: relu, tanh, gelu, sigmoid, sine, leaky_relu
- 5 depths: 1-5 hidden layers
- Fixed width: 8 neurons per layer
- 4 datasets: xor, moons, circles, spirals
- 20 trials per configuration

Total: 6 × 5 × 4 × 20 = 2,400 experiments
Expected runtime: ~15 minutes on 8-core machine
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.jobs import JobManager
from src.core.persistence import ExperimentStore
from src.analysis.config_selection import Phase3Config, estimate_runtime


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'

    # Initialize store and job manager
    store = ExperimentStore(str(data_path))
    job_manager = JobManager(store, n_workers=None)  # Auto-detect workers

    # Get Phase 3 configuration
    config = Phase3Config()
    element_configs = config.get_element_configs()
    summary = config.get_summary()

    # Estimate runtime
    runtime = estimate_runtime(
        n_experiments=summary['total_experiments'],
        n_workers=job_manager.n_workers
    )

    print("=" * 60)
    print("NEURAL ELEMENTS - Phase 3: Statistical Robustness")
    print("=" * 60)

    print("\n📊 Experiment Matrix:")
    print(f"   Activations: {', '.join(summary['activations'])}")
    print(f"   Depths: {summary['depths']}")
    print(f"   Width: {summary['width']} neurons/layer")
    print(f"   Datasets: {', '.join(summary['datasets'])}")
    print(f"   Trials per config: {summary['n_trials']}")

    print(f"\n📈 Scale:")
    print(f"   Element configurations: {summary['n_element_configs']}")
    print(f"   Total experiments: {summary['total_experiments']:,}")

    print(f"\n⚙️  Resources:")
    print(f"   Workers: {job_manager.n_workers}")
    print(f"   Est. time: {runtime['total_minutes']:.1f} minutes")
    print(f"   Est. rate: {runtime['experiments_per_second']:.1f} exp/sec")

    print(f"\n🔧 Training Config:")
    for key, value in summary['training_config'].items():
        print(f"   {key}: {value}")

    # Show sample configs
    print("\n📋 Sample configurations:")
    for i, cfg in enumerate(element_configs[:5]):
        layers = cfg['hidden_layers']
        activation = cfg['activation']
        print(f"   {i+1}. {activation.upper()}-{len(layers)}×{layers[0]}")
    print(f"   ... and {len(element_configs) - 5} more")

    # Confirm
    print("\n" + "-" * 60)
    response = input("Start Phase 3 training? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    print("\n🚀 Submitting job...")

    # Submit the job
    job_id = job_manager.submit_bulk_job(
        element_configs=element_configs,
        dataset_names=config.datasets,
        training_config=config.training_config,
        n_trials=config.n_trials,
    )

    print(f"\n✅ Job submitted: {job_id}")
    print(f"   Total runs: {summary['total_experiments']:,}")

    print("\n📡 Monitor progress:")
    print(f"   curl http://localhost:5000/api/bulk/jobs/{job_id}")
    print("   Or check the web UI at http://localhost:5000")

    # Option to wait
    print("\n" + "-" * 60)
    wait = input("Wait for completion? [y/N]: ").strip().lower()
    if wait == 'y':
        print("\n⏳ Waiting for job to complete...")
        print("   (Press Ctrl+C to stop waiting - job continues in background)\n")

        start_time = time.time()

        try:
            while True:
                job = job_manager.get_job_status(job_id)
                if job is None:
                    print("\n❌ Job not found!")
                    break

                progress = job.completed_runs + job.failed_runs
                pct = 100 * progress / job.total_runs if job.total_runs > 0 else 0
                elapsed = time.time() - start_time

                # Calculate rate and ETA
                rate = progress / elapsed if elapsed > 0 else 0
                remaining = job.total_runs - progress
                eta = remaining / rate if rate > 0 else 0

                status_icon = {
                    'running': '🔄',
                    'completed': '✅',
                    'failed': '❌',
                    'cancelled': '🛑',
                }.get(job.status.value, '⏳')

                print(f"\r{status_icon} Progress: {progress:,}/{job.total_runs:,} ({pct:.1f}%) "
                      f"| ✓ {job.completed_runs:,} ✗ {job.failed_runs} "
                      f"| {rate:.1f}/s | ETA: {eta:.0f}s   ",
                      end='', flush=True)

                if job.status.value in ('completed', 'failed', 'cancelled'):
                    elapsed_min = elapsed / 60
                    print(f"\n\n{'✅' if job.status.value == 'completed' else '❌'} "
                          f"Job {job.status.value} in {elapsed_min:.1f} minutes")
                    print(f"   Completed: {job.completed_runs:,}")
                    print(f"   Failed: {job.failed_runs}")
                    break

                time.sleep(2)

        except KeyboardInterrupt:
            print("\n\n⏸️  Stopped waiting. Job continues in background.")

    print("\n" + "=" * 60)
    print("Phase 3 job submitted. Next steps:")
    print("  1. Wait for experiments to complete")
    print("  2. Run aggregation: python examples/aggregate_phase3.py")
    print("  3. Generate visualizations: python examples/visualize_phase3.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
