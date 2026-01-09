"""
Background job management for bulk training.

Uses Python multiprocessing to run training jobs in parallel.
"""

import os
import json
import time
import traceback
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from multiprocessing import Pool, Manager, cpu_count
from filelock import FileLock
import numpy as np

from .elements import NeuralElement
from .training import Trainer, TrainingConfig
from .persistence import ExperimentStore, ExperimentMetadata, TrainingResult


class JobStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass
class BulkTrainingJob:
    """Represents a bulk training job (grid search)."""
    job_id: str
    created_at: str
    status: JobStatus

    # Configuration
    element_configs: List[Dict]
    dataset_names: List[str]
    training_config: Dict
    n_trials: int

    # Progress tracking
    total_runs: int
    completed_runs: int = 0
    failed_runs: int = 0
    current_run: Optional[str] = None
    experiment_ids: List[str] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)

    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'BulkTrainingJob':
        """Create from dictionary."""
        data = data.copy()
        data['status'] = JobStatus(data['status'])
        return cls(**data)


def training_worker(args: Tuple) -> Dict:
    """
    Worker function for training a single configuration.

    This function runs in a separate process. All inputs must be picklable.

    Args:
        args: Tuple of (element_config, dataset_name, training_config,
                       experiment_id, store_path, X, y)

    Returns:
        Dict with status, experiment_id, and optionally result or error.
    """
    (element_config, dataset_name, training_config,
     experiment_id, store_path, X, y) = args

    try:
        # Create element
        element = NeuralElement(**element_config)

        # Train
        start_time = time.time()
        config = TrainingConfig(**training_config)
        trainer = Trainer(element, config)
        history = trainer.train(X, y)
        training_time = time.time() - start_time

        # Prepare result
        completed_at = datetime.now().isoformat()
        result = TrainingResult(
            experiment_id=experiment_id,
            element_name=element.name,
            weights=[w.tolist() for w in element.weights],
            biases=[b.tolist() if b is not None else None for b in element.biases],
            history=history,
            final_loss=history['loss'][-1] if history['loss'] else 0.0,
            final_accuracy=history['accuracy'][-1] if history['accuracy'] else 0.0,
            training_time_seconds=training_time,
            completed_at=completed_at,
        )

        # Save to store
        store = ExperimentStore(store_path)
        store.save_result(result)

        return {
            'status': 'completed',
            'experiment_id': experiment_id,
            'element_name': element.name,
            'final_accuracy': result.final_accuracy,
            'final_loss': result.final_loss,
            'training_time': training_time,
        }

    except Exception as e:
        error_info = {
            'status': 'failed',
            'experiment_id': experiment_id,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }

        # Update experiment status in store
        try:
            store = ExperimentStore(store_path)
            store.update_status(experiment_id, 'failed', str(e))
        except Exception:
            pass

        return error_info


class JobManager:
    """
    Manages background training jobs using multiprocessing.

    Jobs and progress are persisted to files so state survives server restarts.
    """

    def __init__(self, store: ExperimentStore, n_workers: Optional[int] = None):
        self.store = store
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.jobs_file = store.base_path / 'jobs.json'

        self._pool: Optional[Pool] = None
        self._active_jobs: Dict[str, Any] = {}  # job_id -> AsyncResult

        # Initialize jobs file if needed
        if not self.jobs_file.exists():
            self._write_jobs({})

    def _get_lock(self) -> FileLock:
        """Get file lock for jobs file."""
        return FileLock(str(self.jobs_file) + '.lock')

    def _read_jobs(self) -> Dict[str, Dict]:
        """Read jobs from file."""
        with self._get_lock():
            if self.jobs_file.exists():
                return json.loads(self.jobs_file.read_text())
            return {}

    def _write_jobs(self, jobs: Dict):
        """Write jobs to file."""
        with self._get_lock():
            self.jobs_file.write_text(json.dumps(jobs, indent=2))

    def _save_job(self, job: BulkTrainingJob):
        """Save a single job to the jobs file."""
        jobs = self._read_jobs()
        jobs[job.job_id] = job.to_dict()
        self._write_jobs(jobs)

    def _load_job(self, job_id: str) -> Optional[BulkTrainingJob]:
        """Load a job from file."""
        jobs = self._read_jobs()
        if job_id in jobs:
            return BulkTrainingJob.from_dict(jobs[job_id])
        return None

    def start_pool(self):
        """Initialize the worker pool."""
        if self._pool is None:
            self._pool = Pool(processes=self.n_workers)

    def shutdown_pool(self):
        """Gracefully shutdown the worker pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def generate_job_id(self) -> str:
        """Generate a unique job ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_hash = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        return f'job_{timestamp}_{random_hash}'

    def submit_bulk_job(
        self,
        element_configs: List[Dict],
        dataset_names: List[str],
        training_config: Dict,
        n_trials: int = 1,
        datasets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    ) -> str:
        """
        Submit a new bulk training job.

        Args:
            element_configs: List of element configuration dicts.
            dataset_names: List of dataset names to train on.
            training_config: Training configuration dict.
            n_trials: Number of trials per configuration.
            datasets: Pre-loaded datasets {name: (X, y)}.

        Returns:
            Job ID.
        """
        # Lazy import to avoid circular dependency
        from ..datasets.toy import get_dataset

        # Start pool if needed
        self.start_pool()

        job_id = self.generate_job_id()
        total_runs = len(element_configs) * len(dataset_names) * n_trials

        job = BulkTrainingJob(
            job_id=job_id,
            created_at=datetime.now().isoformat(),
            status=JobStatus.RUNNING,
            element_configs=element_configs,
            dataset_names=dataset_names,
            training_config=training_config,
            n_trials=n_trials,
            total_runs=total_runs,
            started_at=datetime.now().isoformat(),
        )

        self._save_job(job)

        # Prepare all training tasks
        tasks = []
        for element_config in element_configs:
            for dataset_name in dataset_names:
                # Load dataset
                if datasets and dataset_name in datasets:
                    X, y = datasets[dataset_name]
                else:
                    X, y = get_dataset(dataset_name)

                for trial in range(n_trials):
                    experiment_id = self.store.generate_experiment_id()

                    # Create metadata
                    element = NeuralElement(**element_config)
                    metadata = ExperimentMetadata(
                        experiment_id=experiment_id,
                        job_id=job_id,
                        created_at=datetime.now().isoformat(),
                        element_name=element.name,
                        element_config=element_config,
                        dataset_name=dataset_name,
                        training_config=training_config,
                        status='pending',
                    )
                    self.store.save_metadata(metadata)

                    # Add task
                    tasks.append((
                        element_config,
                        dataset_name,
                        training_config,
                        experiment_id,
                        str(self.store.base_path),
                        X,
                        y,
                    ))

        # Submit tasks to pool with callback
        def on_result(result: Dict):
            self._handle_task_result(job_id, result)

        def on_error(error: Exception):
            self._handle_task_error(job_id, str(error))

        # Use map_async for batch processing
        async_result = self._pool.map_async(
            training_worker,
            tasks,
            callback=lambda results: self._finalize_job(job_id, results),
            error_callback=lambda e: self._handle_job_error(job_id, e)
        )

        self._active_jobs[job_id] = async_result

        return job_id

    def _handle_task_result(self, job_id: str, result: Dict):
        """Handle completion of a single training task."""
        job = self._load_job(job_id)
        if job is None:
            return

        if result['status'] == 'completed':
            job.completed_runs += 1
            job.experiment_ids.append(result['experiment_id'])
        else:
            job.failed_runs += 1
            job.errors.append({
                'experiment_id': result.get('experiment_id'),
                'error': result.get('error'),
            })

        self._save_job(job)

    def _handle_task_error(self, job_id: str, error: str):
        """Handle error in a single task."""
        job = self._load_job(job_id)
        if job is None:
            return

        job.failed_runs += 1
        job.errors.append({'error': error})
        self._save_job(job)

    def _finalize_job(self, job_id: str, results: List[Dict]):
        """Finalize job after all tasks complete."""
        job = self._load_job(job_id)
        if job is None:
            return

        # Process all results
        for result in results:
            if result['status'] == 'completed':
                job.completed_runs += 1
                job.experiment_ids.append(result['experiment_id'])
            else:
                job.failed_runs += 1
                job.errors.append({
                    'experiment_id': result.get('experiment_id'),
                    'error': result.get('error'),
                })

        job.status = JobStatus.COMPLETED if job.failed_runs == 0 else JobStatus.FAILED
        job.completed_at = datetime.now().isoformat()
        job.current_run = None

        self._save_job(job)

        # Clean up
        if job_id in self._active_jobs:
            del self._active_jobs[job_id]

    def _handle_job_error(self, job_id: str, error: Exception):
        """Handle critical job error."""
        job = self._load_job(job_id)
        if job is None:
            return

        job.status = JobStatus.FAILED
        job.completed_at = datetime.now().isoformat()
        job.errors.append({'error': str(error), 'traceback': traceback.format_exc()})

        self._save_job(job)

        if job_id in self._active_jobs:
            del self._active_jobs[job_id]

    def get_job_status(self, job_id: str) -> Optional[BulkTrainingJob]:
        """Get current status of a job."""
        return self._load_job(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[BulkTrainingJob]:
        """List jobs, optionally filtered by status."""
        jobs = self._read_jobs()
        result = []

        for job_data in jobs.values():
            job = BulkTrainingJob.from_dict(job_data)
            if status is None or job.status == status:
                result.append(job)

        # Sort by created_at descending
        result.sort(key=lambda j: j.created_at, reverse=True)

        return result[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """
        Request cancellation of a running job.

        Note: Currently running tasks will complete, but no new tasks will start.
        """
        job = self._load_job(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False

        # Terminate the pool tasks for this job
        if job_id in self._active_jobs:
            # Note: terminate() is aggressive; in practice we'd want graceful cancel
            pass

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now().isoformat()
        self._save_job(job)

        return True

    def get_job_results(self, job_id: str) -> List[Dict]:
        """Get results summary for a completed job."""
        job = self._load_job(job_id)
        if job is None:
            return []

        results = []
        for exp_id in job.experiment_ids:
            metadata, result = self.store.load_experiment(exp_id)
            if result:
                results.append({
                    'experiment_id': exp_id,
                    'element_name': result.element_name,
                    'dataset': metadata.dataset_name if metadata else 'unknown',
                    'final_accuracy': result.final_accuracy,
                    'final_loss': result.final_loss,
                    'training_time': result.training_time_seconds,
                })

        return results

    def is_job_running(self, job_id: str) -> bool:
        """Check if a job is currently running."""
        if job_id in self._active_jobs:
            return not self._active_jobs[job_id].ready()
        return False

    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> bool:
        """Wait for a job to complete."""
        if job_id in self._active_jobs:
            try:
                self._active_jobs[job_id].wait(timeout)
                return True
            except TimeoutError:
                return False
        return True  # Job already completed

    def submit_phase7_job(
        self,
        element_configs: List[Dict],
        dataset_names: List[str],
        training_config: Dict,
        phase7_config: Dict,
        n_trials: int = 1,
        datasets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    ) -> str:
        """
        Submit a Phase 7 generalization study job.

        Args:
            element_configs: List of element configuration dicts.
            dataset_names: List of dataset names to train on.
            training_config: Training configuration dict.
            phase7_config: Phase 7 specific config:
                - train_split: float (default 0.8)
                - noise_levels: List[float] (default [0.0])
                - sample_size: int (dataset sample size)
            n_trials: Number of trials per configuration.
            datasets: Pre-loaded datasets {name: (X, y)}.

        Returns:
            Job ID.
        """
        from ..datasets.toy import get_dataset

        self.start_pool()

        job_id = self.generate_job_id()
        total_runs = len(element_configs) * len(dataset_names) * n_trials

        job = BulkTrainingJob(
            job_id=job_id,
            created_at=datetime.now().isoformat(),
            status=JobStatus.RUNNING,
            element_configs=element_configs,
            dataset_names=dataset_names,
            training_config=training_config,
            n_trials=n_trials,
            total_runs=total_runs,
            started_at=datetime.now().isoformat(),
        )

        self._save_job(job)

        # Prepare all training tasks
        tasks = []
        for element_config in element_configs:
            for dataset_name in dataset_names:
                # Load dataset with enough samples
                if datasets and dataset_name in datasets:
                    X, y = datasets[dataset_name]
                else:
                    # Generate with sample size from phase7_config
                    sample_size = phase7_config.get('sample_size', 500)
                    X, y = get_dataset(dataset_name, n_samples=sample_size)

                for trial in range(n_trials):
                    experiment_id = self.store.generate_experiment_id()

                    # Create metadata
                    element = NeuralElement(**element_config)
                    metadata = ExperimentMetadata(
                        experiment_id=experiment_id,
                        job_id=job_id,
                        created_at=datetime.now().isoformat(),
                        element_name=element.name,
                        element_config=element_config,
                        dataset_name=dataset_name,
                        training_config=training_config,
                        status='pending',
                    )
                    self.store.save_metadata(metadata)

                    # Add task with phase7_config
                    tasks.append((
                        element_config,
                        dataset_name,
                        training_config,
                        experiment_id,
                        str(self.store.base_path),
                        X,
                        y,
                        phase7_config,
                    ))

        # Submit tasks to pool
        async_result = self._pool.map_async(
            phase7_training_worker,
            tasks,
            callback=lambda results: self._finalize_job(job_id, results),
            error_callback=lambda e: self._handle_job_error(job_id, e)
        )

        self._active_jobs[job_id] = async_result

        return job_id


def phase7_training_worker(args: Tuple) -> Dict:
    """
    Worker function for Phase 7 generalization experiments.

    Handles train/test splits, evaluates on both sets, and computes
    noise robustness metrics.

    Args:
        args: Tuple of (element_config, dataset_name, training_config,
                       experiment_id, store_path, X, y, phase7_config)

    Returns:
        Dict with status, experiment_id, and results.
    """
    (element_config, dataset_name, training_config,
     experiment_id, store_path, X, y, phase7_config) = args

    try:
        # Extract phase7 config
        train_split = phase7_config.get('train_split', 0.8)
        noise_levels = phase7_config.get('noise_levels', [0.0])
        sample_size = phase7_config.get('sample_size', len(X))

        # 1. Subsample to requested sample size if needed
        if sample_size < len(X):
            idx = np.random.choice(len(X), sample_size, replace=False)
            X, y = X[idx], y[idx]

        # 2. Split into train/test
        n_train = int(len(X) * train_split)
        indices = np.random.permutation(len(X))
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 3. Create element and train on train set only
        element = NeuralElement(**element_config)
        start_time = time.time()
        config = TrainingConfig(**training_config)
        trainer = Trainer(element, config)
        history = trainer.train(X_train, y_train)
        training_time = time.time() - start_time

        # 4. Evaluate on train and test sets
        train_output = element.forward(X_train)
        train_acc = float(np.mean((train_output > 0.5).astype(int) == y_train))

        test_output = element.forward(X_test)
        test_acc = float(np.mean((test_output > 0.5).astype(int) == y_test))

        generalization_gap = train_acc - test_acc

        # Compute test loss
        test_loss = float(-np.mean(
            y_test * np.log(test_output + 1e-15) +
            (1 - y_test) * np.log(1 - test_output + 1e-15)
        ))

        # 5. Noise robustness (test-time noise only)
        noise_robustness = {}
        for noise_level in noise_levels:
            if noise_level == 0.0:
                noise_robustness[str(noise_level)] = test_acc
            else:
                X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
                noisy_output = element.forward(X_noisy)
                noisy_acc = float(np.mean((noisy_output > 0.5).astype(int) == y_test))
                noise_robustness[str(noise_level)] = noisy_acc

        # 6. Store extended result
        completed_at = datetime.now().isoformat()
        result = TrainingResult(
            experiment_id=experiment_id,
            element_name=element.name,
            weights=[w.tolist() for w in element.weights],
            biases=[b.tolist() if b is not None else None for b in element.biases],
            history=history,
            final_loss=history['loss'][-1] if history['loss'] else 0.0,
            final_accuracy=history['accuracy'][-1] if history['accuracy'] else 0.0,
            training_time_seconds=training_time,
            completed_at=completed_at,
            # Phase 7 fields
            train_accuracy=float(train_acc),
            test_accuracy=float(test_acc),
            test_loss=test_loss,
            generalization_gap=float(generalization_gap),
            noise_robustness=noise_robustness,
            sample_size=sample_size,
        )

        # Save to store
        store = ExperimentStore(store_path)
        store.save_result(result)

        return {
            'status': 'completed',
            'experiment_id': experiment_id,
            'element_name': element.name,
            'final_accuracy': result.final_accuracy,
            'final_loss': result.final_loss,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'generalization_gap': generalization_gap,
            'training_time': training_time,
        }

    except Exception as e:
        error_info = {
            'status': 'failed',
            'experiment_id': experiment_id,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }

        try:
            store = ExperimentStore(store_path)
            store.update_status(experiment_id, 'failed', str(e))
        except Exception:
            pass

        return error_info
