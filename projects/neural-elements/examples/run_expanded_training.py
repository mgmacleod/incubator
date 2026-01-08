#!/usr/bin/env python3
"""
Expanded Bulk Training Run - Phase 2

This script runs an expanded grid search to:
1. Add missing activations (sine, swish, linear)
2. Test wider networks (16 neurons)
3. Test deeper networks (4-5 layers)
4. Increase trials for statistical significance

Expected: ~4000 experiments
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.jobs import JobManager
from src.core.persistence import ExperimentStore


def generate_element_configs():
    """Generate all element configurations for the expanded run."""

    configs = []

    # === SECTION 1: Missing Activations ===
    # Test sine, swish, and linear at standard configurations
    new_activations = ['sine', 'swish', 'linear']
    for activation in new_activations:
        for depth in [1, 2, 3]:
            for width in [4, 8]:
                configs.append({
                    'hidden_layers': [width] * depth,
                    'activation': activation,
                })

    # Also add leaky_relu (though our implementation uses default alpha)
    for depth in [1, 2, 3]:
        for width in [4, 8]:
            configs.append({
                'hidden_layers': [width] * depth,
                'activation': 'leaky_relu',
            })

    # === SECTION 2: Wider Networks ===
    # Test width=16 for all existing activations
    existing_activations = ['relu', 'tanh', 'gelu', 'sigmoid']
    for activation in existing_activations:
        for depth in [1, 2, 3]:
            configs.append({
                'hidden_layers': [16] * depth,
                'activation': activation,
            })

    # === SECTION 3: Deeper Networks ===
    # Test depth=4,5 for all activations at width=4
    all_activations = ['relu', 'tanh', 'gelu', 'sigmoid', 'sine', 'swish']
    for activation in all_activations:
        for depth in [4, 5]:
            configs.append({
                'hidden_layers': [4] * depth,
                'activation': activation,
            })

    # === SECTION 4: Asymmetric Architectures ===
    # Test whether shape matters beyond parameter count
    asymmetric_shapes = [
        [8, 4],       # Contracting
        [4, 8],       # Expanding
        [8, 4, 2],    # Funnel
        [2, 4, 8],    # Reverse funnel
        [8, 4, 8],    # Hourglass
        [4, 8, 4],    # Diamond
    ]
    for shape in asymmetric_shapes:
        for activation in ['relu', 'tanh', 'gelu']:
            configs.append({
                'hidden_layers': shape,
                'activation': activation,
            })

    return configs


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'

    # Initialize store and job manager
    store = ExperimentStore(str(data_path))
    job_manager = JobManager(store, n_workers=None)  # Auto-detect workers

    # Generate configs
    element_configs = generate_element_configs()

    # Datasets - focus on the most informative ones
    datasets = [
        'spirals',    # Hard - needs depth
        'xor',        # Classic - simple nonlinearity test
        'moons',      # Medium - curved boundary
        'circles',    # Radial - tests different geometry
    ]

    # Training config
    training_config = {
        'epochs': 500,
        'learning_rate': 0.1,
        'record_every': 50,
    }

    # Number of trials per config
    n_trials = 5

    # Calculate totals
    total_runs = len(element_configs) * len(datasets) * n_trials

    print("=" * 60)
    print("NEURAL ELEMENTS - Expanded Bulk Training Run")
    print("=" * 60)
    print(f"\nConfigurations: {len(element_configs)}")
    print(f"Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"Trials per config: {n_trials}")
    print(f"Total experiments: {total_runs}")
    print(f"\nWorkers: {job_manager.n_workers}")

    # Show sample of configs
    print("\nSample configurations:")
    for i, config in enumerate(element_configs[:5]):
        print(f"  {i+1}. {config}")
    print(f"  ... and {len(element_configs) - 5} more")

    # Confirm
    print("\n" + "-" * 60)
    response = input("Start training? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    print("\nSubmitting job...")

    # Submit the job
    job_id = job_manager.submit_bulk_job(
        element_configs=element_configs,
        dataset_names=datasets,
        training_config=training_config,
        n_trials=n_trials,
    )

    print(f"\nJob submitted: {job_id}")
    print(f"Total runs: {total_runs}")
    print("\nMonitor progress with:")
    print(f"  curl http://localhost:5000/api/bulk/jobs/{job_id}")
    print("\nOr check the web UI at http://localhost:5000")

    # Option to wait
    print("\n" + "-" * 60)
    wait = input("Wait for completion? [y/N]: ").strip().lower()
    if wait == 'y':
        print("\nWaiting for job to complete...")
        print("(Press Ctrl+C to stop waiting - job will continue in background)\n")

        try:
            import time
            while True:
                job = job_manager.get_job_status(job_id)
                if job is None:
                    print("Job not found!")
                    break

                progress = job.completed_runs + job.failed_runs
                pct = 100 * progress / job.total_runs if job.total_runs > 0 else 0

                print(f"\rProgress: {progress}/{job.total_runs} ({pct:.1f}%) "
                      f"- Completed: {job.completed_runs}, Failed: {job.failed_runs}",
                      end='', flush=True)

                if job.status.value in ('completed', 'failed', 'cancelled'):
                    print(f"\n\nJob {job.status.value}!")
                    break

                time.sleep(2)

        except KeyboardInterrupt:
            print("\n\nStopped waiting. Job continues in background.")

    print("\nDone!")


if __name__ == '__main__':
    main()
