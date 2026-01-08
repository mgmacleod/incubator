#!/usr/bin/env python3
"""
Train pending experiments directly (synchronous, single-process).

This is a simpler approach that processes experiments one-by-one,
which is more reliable than multiprocessing for this use case.
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.elements import NeuralElement
from src.core.training import Trainer, TrainingConfig
from src.core.persistence import ExperimentStore, TrainingResult
from src.datasets.toy import get_dataset
from datetime import datetime


def train_pending_experiments(store: ExperimentStore, limit: int = None):
    """Train all pending experiments."""

    # Get pending experiments
    pending = store.list_experiments(status='pending', limit=limit or 10000)
    total = len(pending)

    if total == 0:
        print("No pending experiments found.")
        return

    print(f"Found {total} pending experiments")
    print("-" * 60)

    # Cache datasets
    dataset_cache = {}

    completed = 0
    failed = 0
    start_time = time.time()

    for i, exp_info in enumerate(pending):
        exp_id = exp_info['experiment_id']

        try:
            # Load full metadata
            metadata = store.load_metadata(exp_id)
            if metadata is None:
                print(f"[{i+1}/{total}] {exp_id}: metadata not found, skipping")
                failed += 1
                continue

            # Get dataset (cached)
            dataset_name = metadata.dataset_name
            if dataset_name not in dataset_cache:
                dataset_cache[dataset_name] = get_dataset(dataset_name)
            X, y = dataset_cache[dataset_name]

            # Create element
            element = NeuralElement(**metadata.element_config)

            # Train
            train_start = time.time()
            config = TrainingConfig(**metadata.training_config)
            trainer = Trainer(element, config)
            history = trainer.train(X, y)
            training_time = time.time() - train_start

            # Save result
            result = TrainingResult(
                experiment_id=exp_id,
                element_name=element.name,
                weights=[w.tolist() for w in element.weights],
                biases=[b.tolist() if b is not None else None for b in element.biases],
                history=history,
                final_loss=history['loss'][-1] if history['loss'] else 0.0,
                final_accuracy=history['accuracy'][-1] if history['accuracy'] else 0.0,
                training_time_seconds=training_time,
                completed_at=datetime.now().isoformat(),
            )
            store.save_result(result)

            completed += 1

            # Progress update
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0

            print(f"[{i+1}/{total}] {element.name:15} on {dataset_name:12} "
                  f"acc={result.final_accuracy:.3f} "
                  f"({rate:.1f}/s, ETA: {eta:.0f}s)")

        except Exception as e:
            failed += 1
            store.update_status(exp_id, 'failed', str(e))
            print(f"[{i+1}/{total}] {exp_id}: FAILED - {e}")

    # Summary
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Completed: {completed}, Failed: {failed}")
    print(f"Total time: {elapsed:.1f}s ({completed/elapsed:.1f} experiments/sec)")


def main():
    project_root = Path(__file__).parent.parent
    store = ExperimentStore(str(project_root / 'data'))

    # Get count first
    pending = store.list_experiments(status='pending', limit=10000)
    print(f"\nFound {len(pending)} pending experiments")

    if len(pending) == 0:
        print("Nothing to do!")
        return

    response = input(f"\nTrain all {len(pending)} experiments? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    print()
    train_pending_experiments(store)


if __name__ == '__main__':
    main()
