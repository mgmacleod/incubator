"""
Persistence layer for Neural Elements experiments.

Provides JSON file-based storage for trained models and experiment results.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from filelock import FileLock


@dataclass
class ExperimentMetadata:
    """Metadata for a training experiment."""
    experiment_id: str
    job_id: Optional[str]
    created_at: str
    element_name: str
    element_config: Dict
    dataset_name: str
    training_config: Dict
    status: str = 'pending'  # pending, running, completed, failed
    completed_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TrainingResult:
    """Complete result of a training run."""
    experiment_id: str
    element_name: str
    weights: List[List[List[float]]]
    biases: List[Optional[List[float]]]
    history: Dict[str, List[float]]
    final_loss: float
    final_accuracy: float
    training_time_seconds: float
    completed_at: str

    # Phase 7 generalization fields (optional for backward compatibility)
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_loss: Optional[float] = None
    generalization_gap: Optional[float] = None
    noise_robustness: Optional[Dict[str, float]] = None  # noise_level -> accuracy
    sample_size: Optional[int] = None

    # Phase 8 combination fields (optional for backward compatibility)
    experiment_type: Optional[str] = None  # 'stacking', 'transfer', 'ensemble'

    # Stacking fields
    stack_bottom_accuracy: Optional[float] = None
    stack_bottom_name: Optional[str] = None
    stack_combined_accuracy: Optional[float] = None
    stack_improvement: Optional[float] = None  # combined - bottom

    # Transfer fields
    source_dataset: Optional[str] = None
    target_dataset: Optional[str] = None
    pretrained_accuracy: Optional[float] = None
    scratch_accuracy: Optional[float] = None
    transfer_benefit: Optional[float] = None  # pretrained - scratch
    freeze_mode: Optional[str] = None  # 'freeze_all', 'train_all'

    # Ensemble fields
    ensemble_type: Optional[str] = None  # 'voting', 'averaging', 'weighted'
    ensemble_size: Optional[int] = None
    individual_accuracies: Optional[List[float]] = None
    ensemble_accuracy: Optional[float] = None
    ensemble_improvement: Optional[float] = None  # ensemble - best_individual


class ExperimentStore:
    """
    File-based storage for experiments.

    Storage structure:
        data/
        ├── index.json              # Quick lookup index
        ├── jobs.json               # Job status tracking
        └── experiments/
            └── exp_<timestamp>_<hash>/
                ├── metadata.json   # Config, status, timestamps
                └── result.json     # Weights, biases, history
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.experiments_dir = self.base_path / 'experiments'
        self.index_file = self.base_path / 'index.json'
        self.jobs_file = self.base_path / 'jobs.json'

        # Ensure directories exist
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Initialize index if needed
        if not self.index_file.exists():
            self._write_index({'version': '1.0', 'experiments': {}})

    def _get_lock(self, file_path: Path) -> FileLock:
        """Get a file lock for atomic operations."""
        return FileLock(str(file_path) + '.lock')

    def _read_index(self) -> Dict:
        """Read the index file (caller should hold lock for read-modify-write)."""
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return {'version': '1.0', 'experiments': {}}

    def _write_index(self, index: Dict):
        """Write the index file (caller should hold lock for read-modify-write)."""
        self.index_file.write_text(json.dumps(index, indent=2))

    def _update_index_entry(self, experiment_id: str, updates: Dict):
        """Atomically update a single experiment entry in the index.

        This method holds the lock for the entire read-modify-write cycle
        to prevent race conditions in multiprocessing.
        """
        with self._get_lock(self.index_file):
            index = self._read_index()
            if experiment_id in index['experiments']:
                index['experiments'][experiment_id].update(updates)
            else:
                index['experiments'][experiment_id] = updates
            self._write_index(index)

    def generate_experiment_id(self) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_hash = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        return f'exp_{timestamp}_{random_hash}'

    def _get_experiment_dir(self, experiment_id: str) -> Path:
        """Get the directory for an experiment."""
        return self.experiments_dir / experiment_id

    def save_metadata(self, metadata: ExperimentMetadata):
        """Save experiment metadata."""
        exp_dir = self._get_experiment_dir(metadata.experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = exp_dir / 'metadata.json'
        metadata_file.write_text(json.dumps(asdict(metadata), indent=2))

        # Update index atomically to avoid race conditions in multiprocessing
        self._update_index_entry(metadata.experiment_id, {
            'experiment_id': metadata.experiment_id,
            'element_name': metadata.element_name,
            'dataset': metadata.dataset_name,
            'status': metadata.status,
            'created_at': metadata.created_at,
            'job_id': metadata.job_id,
        })

    def save_result(self, result: TrainingResult):
        """Save training result (weights, biases, history)."""
        exp_dir = self._get_experiment_dir(result.experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        result_file = exp_dir / 'result.json'
        result_file.write_text(json.dumps(asdict(result), indent=2))

        # Update metadata status
        metadata_file = exp_dir / 'metadata.json'
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            metadata['status'] = 'completed'
            metadata['completed_at'] = result.completed_at
            metadata_file.write_text(json.dumps(metadata, indent=2))

        # Update index atomically to avoid race conditions in multiprocessing
        self._update_index_entry(result.experiment_id, {
            'status': 'completed',
            'final_accuracy': result.final_accuracy,
            'final_loss': result.final_loss,
            'training_time': result.training_time_seconds,
        })

    def update_status(self, experiment_id: str, status: str, error: Optional[str] = None):
        """Update experiment status."""
        exp_dir = self._get_experiment_dir(experiment_id)
        metadata_file = exp_dir / 'metadata.json'

        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            metadata['status'] = status
            if error:
                metadata['error'] = error
            if status in ('completed', 'failed'):
                metadata['completed_at'] = datetime.now().isoformat()
            metadata_file.write_text(json.dumps(metadata, indent=2))

        # Update index atomically to avoid race conditions in multiprocessing
        self._update_index_entry(experiment_id, {'status': status})

    def load_metadata(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Load experiment metadata."""
        exp_dir = self._get_experiment_dir(experiment_id)
        metadata_file = exp_dir / 'metadata.json'

        if not metadata_file.exists():
            return None

        data = json.loads(metadata_file.read_text())
        return ExperimentMetadata(**data)

    def load_result(self, experiment_id: str) -> Optional[TrainingResult]:
        """Load training result."""
        exp_dir = self._get_experiment_dir(experiment_id)
        result_file = exp_dir / 'result.json'

        if not result_file.exists():
            return None

        data = json.loads(result_file.read_text())
        return TrainingResult(**data)

    def load_experiment(self, experiment_id: str) -> Tuple[Optional[ExperimentMetadata], Optional[TrainingResult]]:
        """Load both metadata and result for an experiment."""
        return self.load_metadata(experiment_id), self.load_result(experiment_id)

    def list_experiments(
        self,
        status: Optional[str] = None,
        dataset: Optional[str] = None,
        job_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        List experiments with optional filtering.

        Returns index entries (not full metadata) for efficiency.
        """
        index = self._read_index()
        experiments = []

        for exp_id, exp_info in index['experiments'].items():
            # Apply filters
            if status and exp_info.get('status') != status:
                continue
            if dataset and exp_info.get('dataset') != dataset:
                continue
            if job_id and exp_info.get('job_id') != job_id:
                continue

            experiments.append({
                'experiment_id': exp_id,
                **exp_info
            })

        # Sort by created_at descending
        experiments.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # Apply pagination
        return experiments[offset:offset + limit]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its files."""
        exp_dir = self._get_experiment_dir(experiment_id)

        if not exp_dir.exists():
            return False

        # Remove files
        for f in exp_dir.iterdir():
            f.unlink()
        exp_dir.rmdir()

        # Update index
        index = self._read_index()
        if experiment_id in index['experiments']:
            del index['experiments'][experiment_id]
            self._write_index(index)

        return True

    def get_statistics(self) -> Dict:
        """Get aggregate statistics about stored experiments."""
        index = self._read_index()
        experiments = index['experiments']

        stats = {
            'total': len(experiments),
            'by_status': {},
            'by_dataset': {},
            'by_element': {},
        }

        for exp_info in experiments.values():
            # Count by status
            status = exp_info.get('status', 'unknown')
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

            # Count by dataset
            dataset = exp_info.get('dataset', 'unknown')
            stats['by_dataset'][dataset] = stats['by_dataset'].get(dataset, 0) + 1

            # Count by element
            element = exp_info.get('element_name', 'unknown')
            stats['by_element'][element] = stats['by_element'].get(element, 0) + 1

        return stats
