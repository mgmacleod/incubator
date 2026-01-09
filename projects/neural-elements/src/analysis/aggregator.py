"""
Result aggregation for Neural Elements experiments.

Aggregates multiple trials of the same configuration into statistical summaries.
"""

import csv
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from ..core.persistence import ExperimentStore
from .statistics import compute_confidence_interval, ConfidenceInterval
from .config_selection import get_config_key, parse_config_key


@dataclass
class ConfigurationStats:
    """Aggregated statistics for a single configuration across trials."""

    # Configuration identifiers
    config_key: str
    activation: str
    depth: int
    width: int
    dataset: str

    # Accuracy statistics
    mean_accuracy: float
    std_accuracy: float
    ci_lower: float
    ci_upper: float
    min_accuracy: float
    max_accuracy: float

    # Loss statistics
    mean_loss: float
    std_loss: float

    # Training time
    mean_training_time: float

    # Trial counts (with defaults)
    n_trials: int = 0
    n_completed: int = 0
    n_failed: int = 0

    # Optional fields with defaults
    skip_connections: bool = False
    confidence_level: float = 0.95
    experiment_ids: List[str] = field(default_factory=list)


def aggregate_experiments(
    store: ExperimentStore,
    job_id: Optional[str] = None,
    status: str = 'completed',
    confidence: float = 0.95
) -> List[ConfigurationStats]:
    """
    Aggregate experiment results by configuration.

    Args:
        store: ExperimentStore to read from
        job_id: Optional job ID to filter by
        status: Status filter (default 'completed')
        confidence: Confidence level for intervals

    Returns:
        List of ConfigurationStats, one per unique configuration
    """
    # Get all matching experiments
    experiments = store.list_experiments(
        status=status,
        job_id=job_id,
        limit=100000  # Get all
    )

    # Group by configuration
    grouped: Dict[str, Dict[str, Any]] = {}

    for exp in experiments:
        # Parse element name to extract config
        element_name = exp.get('element_name', '')
        dataset = exp.get('dataset', '')

        # Extract activation, depth, width, skip_connections from element name
        # Format is typically: "ACTIVATION-DxW" or "ACTIVATION-DxW-skip"
        config = _parse_element_name(element_name)
        if config is None:
            continue

        activation = config['activation']
        depth = config['depth']
        width = config['width']
        skip_connections = config.get('skip_connections', False)

        # Create config key
        config_key = get_config_key(activation, depth, width, dataset, skip_connections)

        if config_key not in grouped:
            grouped[config_key] = {
                'activation': activation,
                'depth': depth,
                'width': width,
                'dataset': dataset,
                'skip_connections': skip_connections,
                'accuracies': [],
                'losses': [],
                'training_times': [],
                'experiment_ids': [],
                'n_failed': 0,
            }

        # Add this experiment's data
        grouped[config_key]['experiment_ids'].append(exp.get('experiment_id', ''))

        accuracy = exp.get('final_accuracy')
        loss = exp.get('final_loss')
        training_time = exp.get('training_time')

        if accuracy is not None:
            grouped[config_key]['accuracies'].append(accuracy)
        if loss is not None:
            grouped[config_key]['losses'].append(loss)
        if training_time is not None:
            grouped[config_key]['training_times'].append(training_time)

    # Also count failed experiments if we want complete picture
    if job_id:
        failed_experiments = store.list_experiments(
            status='failed',
            job_id=job_id,
            limit=100000
        )
        for exp in failed_experiments:
            element_name = exp.get('element_name', '')
            dataset = exp.get('dataset', '')
            config = _parse_element_name(element_name)
            if config is None:
                continue
            skip_connections = config.get('skip_connections', False)
            config_key = get_config_key(
                config['activation'], config['depth'], config['width'], dataset, skip_connections
            )
            if config_key in grouped:
                grouped[config_key]['n_failed'] += 1

    # Convert to ConfigurationStats
    results = []
    for config_key, data in grouped.items():
        accuracies = data['accuracies']
        losses = data['losses']
        training_times = data['training_times']

        # Compute confidence intervals
        if accuracies:
            acc_ci = compute_confidence_interval(accuracies, confidence)
            mean_accuracy = acc_ci.mean
            std_accuracy = acc_ci.std
            ci_lower = acc_ci.ci_lower
            ci_upper = acc_ci.ci_upper
            min_accuracy = min(accuracies)
            max_accuracy = max(accuracies)
        else:
            mean_accuracy = std_accuracy = ci_lower = ci_upper = float('nan')
            min_accuracy = max_accuracy = float('nan')

        if losses:
            loss_ci = compute_confidence_interval(losses, confidence)
            mean_loss = loss_ci.mean
            std_loss = loss_ci.std
        else:
            mean_loss = std_loss = float('nan')

        mean_training_time = sum(training_times) / len(training_times) if training_times else 0.0

        stats = ConfigurationStats(
            config_key=config_key,
            activation=data['activation'],
            depth=data['depth'],
            width=data['width'],
            dataset=data['dataset'],
            mean_accuracy=mean_accuracy,
            std_accuracy=std_accuracy,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            min_accuracy=min_accuracy,
            max_accuracy=max_accuracy,
            mean_loss=mean_loss,
            std_loss=std_loss,
            mean_training_time=mean_training_time,
            n_trials=len(data['experiment_ids']),
            n_completed=len(accuracies),
            n_failed=data['n_failed'],
            skip_connections=data.get('skip_connections', False),
            confidence_level=confidence,
            experiment_ids=data['experiment_ids'],
        )
        results.append(stats)

    # Sort by activation, depth, skip_connections, dataset for consistent ordering
    results.sort(key=lambda s: (s.activation, s.depth, s.skip_connections, s.dataset))

    return results


def _parse_element_name(element_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse an element name into activation, depth, width, hidden_layers, skip_connections.

    Handles formats like:
    - Uniform: "relu-3x8" -> {'activation': 'relu', 'depth': 3, 'width': 8, 'hidden_layers': [8, 8, 8], ...}
    - Uniform: "leaky_relu-2x4" -> {'activation': 'leaky_relu', 'depth': 2, 'width': 4, ...}
    - Uniform with skip: "relu-3x8-skip" -> {'activation': 'relu', ..., 'skip_connections': True}
    - Non-uniform: "rel-32_8_32" -> {'activation': 'rel', 'depth': 3, 'width': 32, 'hidden_layers': [32, 8, 32], ...}
    - Non-uniform with skip: "sin-4_8_16-skip" -> {..., 'skip_connections': True}
    """
    # Try uniform format first: activation-depthxwidth[-skip]
    # Activation can contain underscores (leaky_relu)
    match = re.match(r'^(.+?)-(\d+)x(\d+)(-skip)?$', element_name)
    if match:
        depth = int(match.group(2))
        width = int(match.group(3))
        return {
            'activation': match.group(1).lower(),
            'depth': depth,
            'width': width,
            'hidden_layers': [width] * depth,
            'skip_connections': match.group(4) is not None,
        }

    # Try non-uniform format: activation-w1_w2_w3[-skip]
    # e.g., "rel-32_8_32" or "sin-4_8_16-skip"
    match = re.match(r'^(.+?)-(\d+(?:_\d+)+)(-skip)?$', element_name)
    if match:
        widths = [int(w) for w in match.group(2).split('_')]
        return {
            'activation': match.group(1).lower(),
            'depth': len(widths),
            'width': max(widths),  # Use max width for grouping
            'hidden_layers': widths,
            'skip_connections': match.group(3) is not None,
        }

    return None


def export_to_csv(
    stats: List[ConfigurationStats],
    filepath: str,
    include_experiment_ids: bool = False
) -> None:
    """
    Export aggregated statistics to CSV.

    Args:
        stats: List of ConfigurationStats to export
        filepath: Path to write CSV file
        include_experiment_ids: Whether to include experiment IDs column
    """
    if not stats:
        return

    # Define columns
    columns = [
        'config_key', 'activation', 'depth', 'width', 'dataset', 'skip_connections',
        'n_trials', 'n_completed', 'n_failed',
        'mean_accuracy', 'std_accuracy', 'ci_lower', 'ci_upper',
        'min_accuracy', 'max_accuracy',
        'mean_loss', 'std_loss',
        'mean_training_time', 'confidence_level'
    ]

    if include_experiment_ids:
        columns.append('experiment_ids')

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for stat in stats:
            row = asdict(stat)
            if not include_experiment_ids:
                del row['experiment_ids']
            else:
                row['experiment_ids'] = ';'.join(row['experiment_ids'])
            writer.writerow(row)


def load_from_csv(filepath: str) -> List[ConfigurationStats]:
    """
    Load aggregated statistics from CSV.

    Args:
        filepath: Path to CSV file

    Returns:
        List of ConfigurationStats
    """
    stats = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types
            experiment_ids = row.get('experiment_ids', '').split(';') if row.get('experiment_ids') else []

            # Handle skip_connections - may be missing in old CSVs
            skip_str = row.get('skip_connections', 'False')
            skip_connections = skip_str.lower() in ('true', '1', 'yes')

            stat = ConfigurationStats(
                config_key=row['config_key'],
                activation=row['activation'],
                depth=int(row['depth']),
                width=int(row['width']),
                dataset=row['dataset'],
                mean_accuracy=float(row['mean_accuracy']),
                std_accuracy=float(row['std_accuracy']),
                ci_lower=float(row['ci_lower']),
                ci_upper=float(row['ci_upper']),
                min_accuracy=float(row['min_accuracy']),
                max_accuracy=float(row['max_accuracy']),
                mean_loss=float(row['mean_loss']),
                std_loss=float(row['std_loss']),
                mean_training_time=float(row['mean_training_time']),
                n_trials=int(row['n_trials']),
                n_completed=int(row['n_completed']),
                n_failed=int(row['n_failed']),
                skip_connections=skip_connections,
                confidence_level=float(row['confidence_level']),
                experiment_ids=experiment_ids,
            )
            stats.append(stat)

    return stats


def get_summary_tables(stats: List[ConfigurationStats]) -> Dict[str, Any]:
    """
    Generate summary tables from aggregated statistics.

    Returns various views of the data useful for analysis.
    """
    if not stats:
        return {}

    # Group by various dimensions
    by_activation: Dict[str, List[ConfigurationStats]] = {}
    by_dataset: Dict[str, List[ConfigurationStats]] = {}
    by_depth: Dict[int, List[ConfigurationStats]] = {}

    for stat in stats:
        by_activation.setdefault(stat.activation, []).append(stat)
        by_dataset.setdefault(stat.dataset, []).append(stat)
        by_depth.setdefault(stat.depth, []).append(stat)

    # Compute activation rankings (overall)
    activation_means = {}
    for activation, group in by_activation.items():
        accuracies = [s.mean_accuracy for s in group if not _isnan(s.mean_accuracy)]
        if accuracies:
            activation_means[activation] = sum(accuracies) / len(accuracies)

    activation_ranking = sorted(
        activation_means.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Best by dataset
    best_by_dataset = {}
    for dataset, group in by_dataset.items():
        valid = [s for s in group if not _isnan(s.mean_accuracy)]
        if valid:
            best = max(valid, key=lambda s: s.mean_accuracy)
            best_by_dataset[dataset] = {
                'activation': best.activation,
                'depth': best.depth,
                'accuracy': best.mean_accuracy,
                'ci': (best.ci_lower, best.ci_upper),
            }

    # Depth effect by activation
    depth_effect = {}
    for activation, group in by_activation.items():
        depth_means = {}
        for stat in group:
            depth_means.setdefault(stat.depth, []).append(stat.mean_accuracy)
        depth_effect[activation] = {
            d: sum(accs) / len(accs)
            for d, accs in depth_means.items()
            if accs and not any(_isnan(a) for a in accs)
        }

    return {
        'activation_ranking': activation_ranking,
        'best_by_dataset': best_by_dataset,
        'depth_effect': depth_effect,
        'total_configs': len(stats),
        'total_experiments': sum(s.n_trials for s in stats),
    }


def _isnan(x: float) -> bool:
    """Check if a value is NaN."""
    return x != x  # NaN is not equal to itself


def print_summary(stats: List[ConfigurationStats]) -> None:
    """Print a formatted summary of aggregated results."""
    summary = get_summary_tables(stats)

    print("=" * 60)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nTotal configurations: {summary['total_configs']}")
    print(f"Total experiments: {summary['total_experiments']}")

    print("\n📊 Activation Ranking (Overall):")
    for i, (activation, mean_acc) in enumerate(summary['activation_ranking'], 1):
        print(f"   {i}. {activation:12s} {mean_acc*100:.1f}%")

    print("\n🎯 Best by Dataset:")
    for dataset, info in summary['best_by_dataset'].items():
        ci_low, ci_high = info['ci']
        print(f"   {dataset:10s} → {info['activation']:12s} "
              f"{info['accuracy']*100:.1f}% "
              f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]")

    print("\n📈 Depth Effect (mean accuracy by depth):")
    for activation, depths in summary['depth_effect'].items():
        depth_str = " → ".join(f"d{d}:{acc*100:.0f}%" for d, acc in sorted(depths.items()))
        print(f"   {activation:12s} {depth_str}")

    print("=" * 60)
