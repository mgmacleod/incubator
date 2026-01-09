"""
Configuration selection for Phase 3: Statistical Robustness.

Defines the focused experiment matrix for Phase 3, which prioritizes
statistical rigor over breadth of coverage.

Design decisions:
- Fixed width=8 for main experiments (avoids width=2 bottleneck issues)
- 20 trials per config (up from 5) for reliable confidence intervals
- 6 core activations (relu, tanh, gelu, sigmoid, sine, leaky_relu)
- 5 depths (1-5 layers)
- 4 datasets (xor, spirals, moons, circles)

Total: 6 activations × 5 depths × 4 datasets × 20 trials = 2,400 experiments
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Phase3Config:
    """Configuration for Phase 3 experiments."""

    # Core activations to test (most informative based on Phase 2)
    activations: List[str] = field(default_factory=lambda: [
        'relu',        # Baseline, widely used
        'tanh',        # Strong performer, bounded
        'gelu',        # Modern, smooth
        'sigmoid',     # Shows depth degradation
        'sine',        # Exceptional on hard tasks
        'leaky_relu',  # Best overall in Phase 2
    ])

    # Depths to test
    depths: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    # Fixed width (8 is the "sweet spot" from Phase 2)
    width: int = 8

    # Datasets (most informative)
    datasets: List[str] = field(default_factory=lambda: [
        'xor',       # Simple nonlinearity test
        'moons',     # Medium difficulty
        'circles',   # Radial boundary
        'spirals',   # Hard, needs depth/capacity
    ])

    # Number of trials for statistical robustness
    n_trials: int = 20

    # Training parameters
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'epochs': 500,
        'learning_rate': 0.1,
        'record_every': 50,
    })

    def get_element_configs(self) -> List[Dict[str, Any]]:
        """Generate all element configurations for Phase 3."""
        configs = []
        for activation in self.activations:
            for depth in self.depths:
                configs.append({
                    'hidden_layers': [self.width] * depth,
                    'activation': activation,
                })
        return configs

    def get_total_experiments(self) -> int:
        """Calculate total number of experiments."""
        n_configs = len(self.activations) * len(self.depths)
        return n_configs * len(self.datasets) * self.n_trials

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            'activations': self.activations,
            'depths': self.depths,
            'width': self.width,
            'datasets': self.datasets,
            'n_trials': self.n_trials,
            'n_element_configs': len(self.activations) * len(self.depths),
            'total_experiments': self.get_total_experiments(),
            'training_config': self.training_config,
        }


def generate_phase3_configs(
    include_width_variants: bool = False
) -> Phase3Config:
    """
    Generate the Phase 3 configuration.

    Args:
        include_width_variants: If True, also include width=4 variants
                               for key activations (adds ~600 experiments)

    Returns:
        Phase3Config with the experiment matrix
    """
    config = Phase3Config()

    if include_width_variants:
        # This would require a modified approach to handle multiple widths
        # For now, we keep it simple with fixed width=8
        pass

    return config


def get_config_key(
    activation: str,
    depth: int,
    width: int,
    dataset: str,
    skip_connections: bool = False
) -> str:
    """
    Generate a unique key for a configuration.

    Used for grouping trials of the same configuration.
    """
    skip_suffix = "_skip" if skip_connections else ""
    return f"{activation}_d{depth}_w{width}_{dataset}{skip_suffix}"


def parse_config_key(key: str) -> Dict[str, Any]:
    """
    Parse a configuration key back into components.

    Args:
        key: Configuration key like "relu_d3_w8_spirals" or "relu_d3_w8_spirals_skip"

    Returns:
        Dictionary with activation, depth, width, dataset, skip_connections
    """
    parts = key.split('_')
    skip_connections = len(parts) > 4 and parts[-1] == 'skip'
    return {
        'activation': parts[0],
        'depth': int(parts[1][1:]),  # Remove 'd' prefix
        'width': int(parts[2][1:]),   # Remove 'w' prefix
        'dataset': parts[3],
        'skip_connections': skip_connections,
    }


@dataclass
class Phase4Config:
    """
    Configuration for Phase 4: Extended Depth Study.

    Tests deeper networks (6-10 layers) and skip connections to answer:
    - Does sigmoid collapse to random chance (50%)?
    - Does sine remain stable at extreme depths?
    - Do skip connections rescue deep networks?

    Total: 4 activations x 4 depths x 4 datasets x 20 trials x 2 variants = 2,560 experiments
    """

    # Focus activations (key behaviors from Phase 3)
    activations: List[str] = field(default_factory=lambda: [
        'sigmoid',     # Shows depth degradation - find the limit
        'relu',        # Baseline control
        'sine',        # Exceptional on hard tasks - test stability
        'tanh',        # Strong performer - control
    ])

    # Extended depths beyond Phase 3
    depths: List[int] = field(default_factory=lambda: [6, 7, 8, 10])

    # Fixed width (same as Phase 3 for comparability)
    width: int = 8

    # Datasets (same as Phase 3)
    datasets: List[str] = field(default_factory=lambda: [
        'xor',       # Simple nonlinearity test
        'moons',     # Medium difficulty
        'circles',   # Radial boundary
        'spirals',   # Hard, needs depth/capacity
    ])

    # Number of trials for statistical robustness
    n_trials: int = 20

    # Skip connections toggle
    skip_connections: bool = False

    # Training parameters (may need adjustment for deeper networks)
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'epochs': 500,
        'learning_rate': 0.1,
        'record_every': 50,
    })

    def get_element_configs(self) -> List[Dict[str, Any]]:
        """Generate all element configurations for Phase 4."""
        configs = []
        for activation in self.activations:
            for depth in self.depths:
                configs.append({
                    'hidden_layers': [self.width] * depth,
                    'activation': activation,
                    'skip_connections': self.skip_connections,
                })
        return configs

    def get_total_experiments(self) -> int:
        """Calculate total number of experiments."""
        n_configs = len(self.activations) * len(self.depths)
        return n_configs * len(self.datasets) * self.n_trials

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            'activations': self.activations,
            'depths': self.depths,
            'width': self.width,
            'datasets': self.datasets,
            'n_trials': self.n_trials,
            'skip_connections': self.skip_connections,
            'n_element_configs': len(self.activations) * len(self.depths),
            'total_experiments': self.get_total_experiments(),
            'training_config': self.training_config,
        }


@dataclass
class Phase5Config:
    """
    Configuration for Phase 5: Architecture Space Exploration.

    Tests non-uniform architectures to understand how width patterns affect learning:
    - Bottleneck architectures (info must pass through narrow layer)
    - Pyramid architectures (expanding, contracting, diamond)
    - Parameter-matched comparisons (same params, different shapes)

    Total: ~11 architectures × 4 activations × 4 datasets × 20 trials = ~3,520 experiments
    """

    # Non-uniform architectures organized by pattern type
    architectures: Dict[str, List[int]] = field(default_factory=lambda: {
        # Bottleneck patterns - info passes through narrow layer
        'bottleneck_severe': [32, 8, 32],
        'bottleneck_moderate': [16, 4, 16],
        'bottleneck_extreme': [8, 2, 8],

        # Pyramid patterns - width changes monotonically
        'pyramid_expanding': [4, 8, 16],
        'pyramid_contracting': [16, 8, 4],
        'pyramid_diamond': [4, 8, 16, 8, 4],

        # Parameter-matched comparisons (~150-200 params)
        'wide_shallow': [32],
        'medium_balanced': [12, 12],
        'narrow_deep': [8, 8, 8],

        # Uniform baselines for comparison
        'uniform_d3': [8, 8, 8],
        'uniform_d5': [8, 8, 8, 8, 8],
    })

    # Best activations from Phase 3/4 (excluding sigmoid which collapses)
    activations: List[str] = field(default_factory=lambda: [
        'relu',
        'sine',
        'tanh',
        'leaky_relu',
    ])

    # Datasets
    datasets: List[str] = field(default_factory=lambda: [
        'xor',
        'moons',
        'circles',
        'spirals',
    ])

    # Number of trials for statistical robustness
    n_trials: int = 20

    # Training parameters
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'epochs': 500,
        'learning_rate': 0.1,
        'record_every': 50,
    })

    def get_element_configs(self) -> List[Dict[str, Any]]:
        """Generate all element configurations for Phase 5."""
        configs = []
        for pattern_name, hidden_layers in self.architectures.items():
            for activation in self.activations:
                configs.append({
                    'hidden_layers': hidden_layers,
                    'activation': activation,
                })
        return configs

    def get_total_experiments(self) -> int:
        """Calculate total number of experiments."""
        n_configs = len(self.architectures) * len(self.activations)
        return n_configs * len(self.datasets) * self.n_trials

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            'architectures': self.architectures,
            'activations': self.activations,
            'datasets': self.datasets,
            'n_trials': self.n_trials,
            'n_element_configs': len(self.architectures) * len(self.activations),
            'total_experiments': self.get_total_experiments(),
            'training_config': self.training_config,
        }


def estimate_runtime(
    n_experiments: int,
    n_workers: int = 8,
    avg_seconds_per_experiment: float = 0.3
) -> Dict[str, float]:
    """
    Estimate runtime for a batch of experiments.

    Based on Phase 2 observations: ~0.3 seconds per experiment on average.

    Args:
        n_experiments: Total number of experiments
        n_workers: Number of parallel workers
        avg_seconds_per_experiment: Average time per experiment

    Returns:
        Dictionary with time estimates
    """
    total_seconds = n_experiments * avg_seconds_per_experiment / n_workers

    return {
        'total_seconds': total_seconds,
        'total_minutes': total_seconds / 60,
        'experiments_per_second': n_workers / avg_seconds_per_experiment,
    }
