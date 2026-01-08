"""
Toy datasets for neural element experimentation.

These datasets are designed to:
1. Be visually interpretable (2D input)
2. Have varying levels of complexity
3. Require different kinds of decision boundaries
4. Train quickly on CPU

Each dataset reveals different aspects of a network's capabilities.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable


def xor_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classic XOR problem - the simplest non-linearly separable dataset.

    Requires at least one hidden layer to solve.
    Tests: Basic nonlinear capability

    Args:
        n_samples: Number of samples
        noise: Standard deviation of Gaussian noise
        seed: Random seed

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate points in four quadrants
    n_per_class = n_samples // 4

    # Class 0: top-left and bottom-right
    X0_tl = np.random.randn(n_per_class, 2) * noise + np.array([-1, 1])
    X0_br = np.random.randn(n_per_class, 2) * noise + np.array([1, -1])

    # Class 1: top-right and bottom-left
    X1_tr = np.random.randn(n_per_class, 2) * noise + np.array([1, 1])
    X1_bl = np.random.randn(n_per_class, 2) * noise + np.array([-1, -1])

    X = np.vstack([X0_tl, X0_br, X1_tr, X1_bl])
    y = np.hstack([
        np.zeros(2 * n_per_class),
        np.ones(2 * n_per_class)
    ])

    # Shuffle
    indices = np.random.permutation(len(y))
    return X[indices], y[indices].astype(int)


def spirals(
    n_samples: int = 200,
    noise: float = 0.2,
    n_turns: float = 1.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two interleaved spirals - a challenging classification problem.

    Requires depth and expressiveness to solve well.
    Tests: Complex nonlinear boundaries, depth benefits

    Args:
        n_samples: Total number of samples
        noise: Noise level
        n_turns: Number of spiral turns
        seed: Random seed

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    n_per_class = n_samples // 2

    # Generate spiral for class 0
    theta0 = np.sqrt(np.random.rand(n_per_class)) * n_turns * 2 * np.pi
    r0 = theta0 / (n_turns * 2 * np.pi) * 2
    X0 = np.column_stack([
        r0 * np.cos(theta0) + np.random.randn(n_per_class) * noise,
        r0 * np.sin(theta0) + np.random.randn(n_per_class) * noise
    ])

    # Generate spiral for class 1 (rotated 180 degrees)
    theta1 = np.sqrt(np.random.rand(n_per_class)) * n_turns * 2 * np.pi
    r1 = theta1 / (n_turns * 2 * np.pi) * 2
    X1 = np.column_stack([
        -r1 * np.cos(theta1) + np.random.randn(n_per_class) * noise,
        -r1 * np.sin(theta1) + np.random.randn(n_per_class) * noise
    ])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    indices = np.random.permutation(len(y))
    return X[indices], y[indices].astype(int)


def moons(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two interleaving half circles (moons).

    A classic dataset requiring nonlinear decision boundary.
    Tests: Curved boundaries

    Args:
        n_samples: Total number of samples
        noise: Noise level
        seed: Random seed

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    n_per_class = n_samples // 2

    # Upper moon
    theta0 = np.linspace(0, np.pi, n_per_class)
    X0 = np.column_stack([
        np.cos(theta0) + np.random.randn(n_per_class) * noise,
        np.sin(theta0) + np.random.randn(n_per_class) * noise
    ])

    # Lower moon (shifted)
    theta1 = np.linspace(0, np.pi, n_per_class)
    X1 = np.column_stack([
        1 - np.cos(theta1) + np.random.randn(n_per_class) * noise,
        0.5 - np.sin(theta1) + np.random.randn(n_per_class) * noise
    ])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    indices = np.random.permutation(len(y))
    return X[indices], y[indices].astype(int)


def circles(
    n_samples: int = 200,
    noise: float = 0.1,
    factor: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two concentric circles.

    Tests: Radial decision boundaries
    Can be solved with distance-based features.

    Args:
        n_samples: Total number of samples
        noise: Noise level
        factor: Ratio between inner and outer circle radii
        seed: Random seed

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    n_per_class = n_samples // 2

    # Outer circle
    theta0 = np.random.rand(n_per_class) * 2 * np.pi
    r0 = 1 + np.random.randn(n_per_class) * noise
    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])

    # Inner circle
    theta1 = np.random.rand(n_per_class) * 2 * np.pi
    r1 = factor + np.random.randn(n_per_class) * noise
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    indices = np.random.permutation(len(y))
    return X[indices], y[indices].astype(int)


def gaussian_clusters(
    n_samples: int = 200,
    n_clusters: int = 4,
    cluster_std: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian clusters arranged in a pattern.

    Alternating class labels create a non-trivial boundary.
    Tests: Multi-region decision boundaries

    Args:
        n_samples: Total number of samples
        n_clusters: Number of clusters (should be even)
        cluster_std: Standard deviation of each cluster
        seed: Random seed

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    n_per_cluster = n_samples // n_clusters

    # Arrange clusters in a circle
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.5

    X_list = []
    y_list = []

    for i, center in enumerate(centers):
        X_cluster = np.random.randn(n_per_cluster, 2) * cluster_std + center
        X_list.append(X_cluster)
        y_list.append(np.full(n_per_cluster, i % 2))

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    indices = np.random.permutation(len(y))
    return X[indices], y[indices].astype(int)


def checkerboard(
    n_samples: int = 200,
    grid_size: int = 2,
    noise: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Checkerboard pattern.

    A challenging pattern requiring complex decision boundaries.
    Tests: Highly non-convex regions

    Args:
        n_samples: Total number of samples
        grid_size: Number of squares per side
        noise: Noise level
        seed: Random seed

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.rand(n_samples, 2) * 2 - 1  # Range [-1, 1]

    # Determine checkerboard position
    x_idx = np.floor((X[:, 0] + 1) * grid_size / 2).astype(int)
    y_idx = np.floor((X[:, 1] + 1) * grid_size / 2).astype(int)

    y = ((x_idx + y_idx) % 2).astype(int)

    if noise > 0:
        X += np.random.randn(n_samples, 2) * noise

    return X, y


def rings(
    n_samples: int = 200,
    n_rings: int = 3,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multiple concentric rings with alternating labels.

    Tests: Multiple radial boundaries

    Args:
        n_samples: Total number of samples
        n_rings: Number of rings
        noise: Noise level
        seed: Random seed

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    n_per_ring = n_samples // n_rings

    X_list = []
    y_list = []

    for i in range(n_rings):
        theta = np.random.rand(n_per_ring) * 2 * np.pi
        r = (i + 1) / n_rings + np.random.randn(n_per_ring) * noise * 0.5
        X_ring = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        X_list.append(X_ring)
        y_list.append(np.full(n_per_ring, i % 2))

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    indices = np.random.permutation(len(y))
    return X[indices], y[indices].astype(int)


def swiss_roll_2d(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D projection of a Swiss roll.

    Tests: Highly curved, spiral-like boundary

    Args:
        n_samples: Total number of samples
        noise: Noise level
        seed: Random seed

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))

    X = np.column_stack([
        t * np.cos(t) + np.random.randn(n_samples) * noise,
        t * np.sin(t) + np.random.randn(n_samples) * noise
    ])

    # Normalize
    X = (X - X.mean(axis=0)) / X.std()

    # Label based on angle
    y = (t > np.median(t)).astype(int)

    return X, y


def linear_separable(
    n_samples: int = 200,
    margin: float = 0.5,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly separable dataset.

    The simplest case - can be solved without hidden layers.
    Tests: Baseline linear capability

    Args:
        n_samples: Total number of samples
        margin: Separation margin between classes
        noise: Noise level
        seed: Random seed

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    n_per_class = n_samples // 2

    # Class 0: below the line y = x - margin
    X0 = np.random.randn(n_per_class, 2) * noise
    X0[:, 1] -= margin

    # Class 1: above the line y = x + margin
    X1 = np.random.randn(n_per_class, 2) * noise
    X1[:, 1] += margin

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    indices = np.random.permutation(len(y))
    return X[indices], y[indices].astype(int)


# Dataset registry
DATASETS: Dict[str, Dict] = {
    'xor': {
        'function': xor_dataset,
        'name': 'XOR',
        'description': 'Classic XOR problem - simplest non-linear dataset',
        'difficulty': 'easy',
        'requires_depth': True,
        'default_params': {'n_samples': 200, 'noise': 0.1},
    },
    'spirals': {
        'function': spirals,
        'name': 'Spirals',
        'description': 'Two interleaved spirals - challenging classification',
        'difficulty': 'hard',
        'requires_depth': True,
        'default_params': {'n_samples': 200, 'noise': 0.2, 'n_turns': 1.5},
    },
    'moons': {
        'function': moons,
        'name': 'Moons',
        'description': 'Two interleaving half circles',
        'difficulty': 'medium',
        'requires_depth': True,
        'default_params': {'n_samples': 200, 'noise': 0.1},
    },
    'circles': {
        'function': circles,
        'name': 'Circles',
        'description': 'Two concentric circles',
        'difficulty': 'medium',
        'requires_depth': True,
        'default_params': {'n_samples': 200, 'noise': 0.1, 'factor': 0.5},
    },
    'gaussian_clusters': {
        'function': gaussian_clusters,
        'name': 'Gaussian Clusters',
        'description': 'Multiple Gaussian clusters with alternating labels',
        'difficulty': 'medium',
        'requires_depth': True,
        'default_params': {'n_samples': 200, 'n_clusters': 4, 'cluster_std': 0.3},
    },
    'checkerboard': {
        'function': checkerboard,
        'name': 'Checkerboard',
        'description': 'Checkerboard pattern - highly non-convex',
        'difficulty': 'hard',
        'requires_depth': True,
        'default_params': {'n_samples': 200, 'grid_size': 2, 'noise': 0.0},
    },
    'rings': {
        'function': rings,
        'name': 'Rings',
        'description': 'Concentric rings with alternating labels',
        'difficulty': 'hard',
        'requires_depth': True,
        'default_params': {'n_samples': 200, 'n_rings': 3, 'noise': 0.1},
    },
    'swiss_roll_2d': {
        'function': swiss_roll_2d,
        'name': 'Swiss Roll 2D',
        'description': '2D projection of Swiss roll manifold',
        'difficulty': 'hard',
        'requires_depth': True,
        'default_params': {'n_samples': 200, 'noise': 0.1},
    },
    'linear': {
        'function': linear_separable,
        'name': 'Linear',
        'description': 'Linearly separable - baseline dataset',
        'difficulty': 'trivial',
        'requires_depth': False,
        'default_params': {'n_samples': 200, 'margin': 0.5, 'noise': 0.1},
    },
}


def get_dataset(
    name: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a dataset by name.

    Args:
        name: Dataset name
        **kwargs: Override default parameters

    Returns:
        X: Features
        y: Labels
    """
    if name not in DATASETS:
        available = ', '.join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    dataset_info = DATASETS[name]
    params = dataset_info['default_params'].copy()
    params.update(kwargs)

    return dataset_info['function'](**params)


def list_datasets() -> Dict[str, Dict]:
    """List all available datasets with their metadata."""
    return {
        name: {k: v for k, v in info.items() if k != 'function'}
        for name, info in DATASETS.items()
    }
