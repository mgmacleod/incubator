"""
Generalization metrics for Phase 7: Generalization Study.

Provides functions for computing:
- Generalization gap (train_acc - test_acc)
- Noise robustness (accuracy on noisy test data)
- Sample efficiency curves
"""

from typing import Dict, List, Tuple, Callable
import numpy as np


def compute_generalization_gap(train_acc: float, test_acc: float) -> float:
    """
    Compute the generalization gap.

    A positive gap indicates overfitting (train > test).
    A negative gap indicates underfitting or lucky test set (rare).

    Args:
        train_acc: Training accuracy (0-1)
        test_acc: Test accuracy (0-1)

    Returns:
        Generalization gap (train_acc - test_acc)
    """
    return train_acc - test_acc


def add_gaussian_noise(X: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    """
    Add Gaussian noise to input data.

    Args:
        X: Input array of shape (n_samples, n_features)
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Noisy input array of same shape
    """
    if seed is not None:
        np.random.seed(seed)

    if noise_level <= 0:
        return X.copy()

    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def compute_noise_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: List[float],
    n_trials: int = 1
) -> Dict[str, float]:
    """
    Compute accuracy at different noise levels.

    Args:
        model: Trained model with accuracy(X, y) method
        X_test: Test features
        y_test: Test labels
        noise_levels: List of noise standard deviations to test
        n_trials: Number of trials to average (for stochastic noise)

    Returns:
        Dictionary mapping noise level (as string) to accuracy
    """
    results = {}

    for noise_level in noise_levels:
        if noise_level == 0.0:
            # No noise - just compute accuracy once
            acc = model.accuracy(X_test, y_test)
        else:
            # Average over multiple noise trials
            accs = []
            for _ in range(n_trials):
                X_noisy = add_gaussian_noise(X_test, noise_level)
                accs.append(model.accuracy(X_noisy, y_test))
            acc = np.mean(accs)

        # Store as string key for JSON serialization
        results[str(noise_level)] = float(acc)

    return results


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Args:
        X: Features array of shape (n_samples, n_features)
        y: Labels array of shape (n_samples,)
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(X)
    n_train = int(n_samples * train_ratio)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample data to a fixed size.

    Args:
        X: Features array
        y: Labels array
        n_samples: Number of samples to keep
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X_subsampled, y_subsampled)
    """
    if seed is not None:
        np.random.seed(seed)

    if n_samples >= len(X):
        return X.copy(), y.copy()

    indices = np.random.choice(len(X), n_samples, replace=False)
    return X[indices], y[indices]


def compute_sample_efficiency(
    train_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    sample_sizes: List[int],
    test_ratio: float = 0.2,
    n_trials: int = 1
) -> Dict[int, float]:
    """
    Compute accuracy at different sample sizes.

    This is useful for understanding sample efficiency - how quickly
    a model reaches good performance with limited data.

    Args:
        train_fn: Function that takes (X_train, y_train, X_test, y_test)
                  and returns test accuracy
        X: Full feature array
        y: Full label array
        sample_sizes: List of sample sizes to test
        test_ratio: Fraction of samples to hold out for testing
        n_trials: Number of trials to average

    Returns:
        Dictionary mapping sample size to accuracy
    """
    results = {}

    for sample_size in sample_sizes:
        accs = []
        for trial in range(n_trials):
            # Subsample to requested size
            X_sub, y_sub = subsample(X, y, sample_size, seed=trial)

            # Split into train/test
            X_train, y_train, X_test, y_test = train_test_split(
                X_sub, y_sub, train_ratio=1-test_ratio, seed=trial
            )

            # Train and evaluate
            acc = train_fn(X_train, y_train, X_test, y_test)
            accs.append(acc)

        results[sample_size] = float(np.mean(accs))

    return results


def aggregate_generalization_metrics(
    experiments: List[Dict],
    group_by: List[str]
) -> Dict[str, Dict]:
    """
    Aggregate generalization metrics by configuration groups.

    Args:
        experiments: List of experiment result dicts with fields:
            - activation, depth, dataset, sample_size
            - train_accuracy, test_accuracy, generalization_gap
            - noise_robustness (dict of noise_level -> accuracy)
        group_by: List of fields to group by (e.g., ['activation', 'depth'])

    Returns:
        Dictionary mapping group key to aggregated metrics
    """
    from collections import defaultdict

    groups = defaultdict(list)

    for exp in experiments:
        # Create group key
        key_parts = [str(exp.get(field, 'unknown')) for field in group_by]
        key = '_'.join(key_parts)
        groups[key].append(exp)

    results = {}
    for key, group_exps in groups.items():
        n = len(group_exps)

        # Aggregate basic metrics
        train_accs = [e.get('train_accuracy', 0) for e in group_exps if e.get('train_accuracy') is not None]
        test_accs = [e.get('test_accuracy', 0) for e in group_exps if e.get('test_accuracy') is not None]
        gaps = [e.get('generalization_gap', 0) for e in group_exps if e.get('generalization_gap') is not None]

        results[key] = {
            'n': n,
            'train_accuracy_mean': np.mean(train_accs) if train_accs else None,
            'train_accuracy_std': np.std(train_accs) if train_accs else None,
            'test_accuracy_mean': np.mean(test_accs) if test_accs else None,
            'test_accuracy_std': np.std(test_accs) if test_accs else None,
            'generalization_gap_mean': np.mean(gaps) if gaps else None,
            'generalization_gap_std': np.std(gaps) if gaps else None,
        }

        # Aggregate noise robustness by noise level
        noise_by_level = defaultdict(list)
        for exp in group_exps:
            noise_dict = exp.get('noise_robustness', {})
            if noise_dict:
                for level, acc in noise_dict.items():
                    noise_by_level[level].append(acc)

        if noise_by_level:
            results[key]['noise_robustness'] = {
                level: {
                    'mean': np.mean(accs),
                    'std': np.std(accs)
                }
                for level, accs in noise_by_level.items()
            }

    return results
