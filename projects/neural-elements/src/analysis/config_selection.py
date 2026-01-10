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


@dataclass
class Phase6Config:
    """
    Configuration for Phase 6: Learning Dynamics Study.

    Studies *how* elements learn by recording gradient flow and weight evolution.
    Tests configurations from Phase 4/5 that showed interesting behaviors.

    Experiment breakdown:
    - Baseline uniform: 4 activations × 4 depths × 4 datasets × 20 trials = 1,280
    - Skip variants: 4 activations × 2 depths × 4 datasets × 20 trials = 640
    - Bottleneck archs: 2 archs × 4 activations × 4 datasets × 20 trials = 640
    Total: ~2,560 experiments
    """

    # Focus on activations that showed interesting behaviors
    activations: List[str] = field(default_factory=lambda: [
        'relu',        # Baseline, hurt by skip connections at depth
        'sigmoid',     # Collapses at depth, rescued by skip
        'sine',        # Stable at extreme depths, hurt by skip
        'tanh',        # Graceful degradation
    ])

    # Depths covering key transition points
    depths: List[int] = field(default_factory=lambda: [1, 3, 5, 8])

    # Fixed width (same as Phase 3-4 for comparability)
    width: int = 8

    # Bottleneck architectures from Phase 5
    bottleneck_architectures: Dict[str, List[int]] = field(default_factory=lambda: {
        'bottleneck_severe': [32, 8, 32],
        'bottleneck_extreme': [8, 2, 8],
    })

    # Datasets
    datasets: List[str] = field(default_factory=lambda: [
        'xor',
        'moons',
        'circles',
        'spirals',
    ])

    # Number of trials for statistical robustness
    n_trials: int = 20

    # Training parameters - more epochs and frequent recording for dynamics
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'epochs': 1000,
        'learning_rate': 0.1,
        'record_every': 10,
        'record_gradients': True,
        'record_weight_stats': True,
    })

    def get_element_configs(self) -> List[Dict[str, Any]]:
        """Generate all element configurations for Phase 6."""
        configs = []

        # 1. Baseline uniform configurations (no skip)
        for activation in self.activations:
            for depth in self.depths:
                configs.append({
                    'hidden_layers': [self.width] * depth,
                    'activation': activation,
                    'skip_connections': False,
                })

        # 2. Skip connection variants (depths 5 and 8 only)
        for activation in self.activations:
            for depth in [5, 8]:
                configs.append({
                    'hidden_layers': [self.width] * depth,
                    'activation': activation,
                    'skip_connections': True,
                })

        # 3. Bottleneck architectures (no skip)
        for arch_name, hidden_layers in self.bottleneck_architectures.items():
            for activation in self.activations:
                configs.append({
                    'hidden_layers': hidden_layers,
                    'activation': activation,
                    'skip_connections': False,
                })

        return configs

    def get_total_experiments(self) -> int:
        """Calculate total number of experiments."""
        n_configs = len(self.get_element_configs())
        return n_configs * len(self.datasets) * self.n_trials

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        configs = self.get_element_configs()

        # Count by type
        baseline_count = len(self.activations) * len(self.depths)
        skip_count = len(self.activations) * 2  # depths 5, 8
        bottleneck_count = len(self.bottleneck_architectures) * len(self.activations)

        return {
            'activations': self.activations,
            'depths': self.depths,
            'width': self.width,
            'bottleneck_architectures': self.bottleneck_architectures,
            'datasets': self.datasets,
            'n_trials': self.n_trials,
            'n_element_configs': len(configs),
            'breakdown': {
                'baseline': baseline_count,
                'skip_variants': skip_count,
                'bottleneck': bottleneck_count,
            },
            'total_experiments': self.get_total_experiments(),
            'training_config': self.training_config,
        }


@dataclass
class Phase7Config:
    """
    Configuration for Phase 7: Generalization Study.

    Tests whether neural elements generalize or just memorize by measuring:
    - Generalization gap (train_acc - test_acc)
    - Noise robustness (accuracy on noisy test data)
    - Sample efficiency (accuracy at different sample sizes)

    Experiment breakdown:
    - 4 activations × 4 depths × 4 datasets × 4 sample_sizes × 20 trials = 5,120 experiments

    Each experiment records train/test accuracy and noise robustness at multiple levels.
    """

    # Focus activations (consistent with Phase 6)
    activations: List[str] = field(default_factory=lambda: [
        'relu',        # Baseline
        'sigmoid',     # Fails at depth - interesting generalization behavior?
        'sine',        # Exceptional - sample efficient?
        'tanh',        # Strong performer
    ])

    # Depths covering key transition points
    depths: List[int] = field(default_factory=lambda: [1, 3, 5, 8])

    # Fixed width (same as previous phases)
    width: int = 8

    # Datasets
    datasets: List[str] = field(default_factory=lambda: [
        'xor',
        'moons',
        'circles',
        'spirals',
    ])

    # Number of trials for statistical robustness
    n_trials: int = 20

    # Phase 7 specific: generalization parameters
    train_split: float = 0.8  # 80% train, 20% test
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])
    sample_sizes: List[int] = field(default_factory=lambda: [50, 100, 200, 500])

    # Training parameters
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'epochs': 1000,
        'learning_rate': 0.1,
        'record_every': 50,
    })

    def get_element_configs(self) -> List[Dict[str, Any]]:
        """Generate all element configurations for Phase 7."""
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
        return n_configs * len(self.datasets) * len(self.sample_sizes) * self.n_trials

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        n_configs = len(self.activations) * len(self.depths)
        return {
            'activations': self.activations,
            'depths': self.depths,
            'width': self.width,
            'datasets': self.datasets,
            'n_trials': self.n_trials,
            'sample_sizes': self.sample_sizes,
            'noise_levels': self.noise_levels,
            'train_split': self.train_split,
            'n_element_configs': n_configs,
            'total_experiments': self.get_total_experiments(),
            'training_config': self.training_config,
        }


@dataclass
class Phase8Config:
    """
    Configuration for Phase 8: Combination & Transfer.

    Tests whether neural element properties are composable through:
    - Stacking: Train element A, freeze, add element B on top
    - Transfer: Train on dataset X, fine-tune on dataset Y
    - Ensembles: Combine predictions from multiple elements

    Experiment breakdown:
    - Stacking: 4 activations × 2 bottom depths × 2 top depths × 4 datasets × 20 trials = 640
    - Transfer: 4 activations × 4 pairs × 2 freeze modes × 20 trials = 640
    - Ensembles: 4 activations × 3 sizes × 3 types × 4 datasets × 5 trials = 720
    - Total: ~2,000 experiments
    """

    # Common parameters
    activations: List[str] = field(default_factory=lambda: [
        'relu',        # Baseline
        'sigmoid',     # Fails at depth - interesting transfer behavior?
        'sine',        # Exceptional - sample efficient?
        'tanh',        # Strong performer
    ])

    datasets: List[str] = field(default_factory=lambda: [
        'xor',
        'moons',
        'circles',
        'spirals',
    ])

    width: int = 8
    n_trials: int = 20

    # Stacking config
    stacking_bottom_depths: List[int] = field(default_factory=lambda: [1, 3])
    stacking_top_depths: List[int] = field(default_factory=lambda: [1, 3])

    # Transfer config
    transfer_pairs: List[tuple] = field(default_factory=lambda: [
        ('xor', 'moons'),
        ('moons', 'xor'),
        ('circles', 'spirals'),
        ('spirals', 'circles'),
    ])
    transfer_freeze_modes: List[str] = field(default_factory=lambda: [
        'freeze_all',   # Feature extraction: freeze all pretrained layers
        'train_all',    # Fine-tuning: train all layers with pretrained init
    ])
    transfer_depth: int = 3  # Fixed depth for transfer experiments

    # Ensemble config
    ensemble_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    ensemble_types: List[str] = field(default_factory=lambda: [
        'voting',     # Majority vote
        'averaging',  # Average probabilities
        'weighted',   # Weight by individual accuracy
    ])
    ensemble_trials: int = 5  # Fewer trials since ensembles train multiple elements

    # Training parameters
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'epochs': 1000,
        'learning_rate': 0.1,
        'record_every': 50,
    })

    def get_stacking_configs(self) -> List[Dict[str, Any]]:
        """Generate all stacking experiment configurations."""
        configs = []
        for activation in self.activations:
            for bottom_depth in self.stacking_bottom_depths:
                for top_depth in self.stacking_top_depths:
                    for dataset in self.datasets:
                        configs.append({
                            'activation': activation,
                            'bottom_depth': bottom_depth,
                            'top_depth': top_depth,
                            'width': self.width,
                            'dataset': dataset,
                        })
        return configs

    def get_transfer_configs(self) -> List[Dict[str, Any]]:
        """Generate all transfer experiment configurations."""
        configs = []
        for activation in self.activations:
            for source, target in self.transfer_pairs:
                for freeze_mode in self.transfer_freeze_modes:
                    configs.append({
                        'activation': activation,
                        'source_dataset': source,
                        'target_dataset': target,
                        'freeze_mode': freeze_mode,
                        'depth': self.transfer_depth,
                        'width': self.width,
                    })
        return configs

    def get_ensemble_configs(self) -> List[Dict[str, Any]]:
        """Generate all ensemble experiment configurations."""
        configs = []
        for activation in self.activations:
            for ensemble_size in self.ensemble_sizes:
                for ensemble_type in self.ensemble_types:
                    for dataset in self.datasets:
                        configs.append({
                            'activation': activation,
                            'ensemble_size': ensemble_size,
                            'ensemble_type': ensemble_type,
                            'dataset': dataset,
                            'depth': self.transfer_depth,  # Use same depth
                            'width': self.width,
                        })
        return configs

    def get_total_experiments(self) -> Dict[str, int]:
        """Calculate total experiments by type."""
        stacking = (len(self.activations) * len(self.stacking_bottom_depths) *
                   len(self.stacking_top_depths) * len(self.datasets) * self.n_trials)
        transfer = (len(self.activations) * len(self.transfer_pairs) *
                   len(self.transfer_freeze_modes) * self.n_trials)
        ensemble = (len(self.activations) * len(self.ensemble_sizes) *
                   len(self.ensemble_types) * len(self.datasets) * self.ensemble_trials)

        return {
            'stacking': stacking,
            'transfer': transfer,
            'ensemble': ensemble,
            'total': stacking + transfer + ensemble,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        totals = self.get_total_experiments()
        return {
            'activations': self.activations,
            'datasets': self.datasets,
            'width': self.width,
            'n_trials': self.n_trials,
            'stacking': {
                'bottom_depths': self.stacking_bottom_depths,
                'top_depths': self.stacking_top_depths,
                'experiments': totals['stacking'],
            },
            'transfer': {
                'pairs': self.transfer_pairs,
                'freeze_modes': self.transfer_freeze_modes,
                'depth': self.transfer_depth,
                'experiments': totals['transfer'],
            },
            'ensemble': {
                'sizes': self.ensemble_sizes,
                'types': self.ensemble_types,
                'trials': self.ensemble_trials,
                'experiments': totals['ensemble'],
            },
            'total_experiments': totals['total'],
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


@dataclass
class Phase9Config:
    """
    Configuration for Phase 9: Evolutionary Discovery.

    Uses genetic algorithms to discover optimal neural element architectures
    beyond grid-based exploration.

    Design decisions:
    - Mixed activations: Per-layer activation support
    - Fitness: Weighted sum (60% accuracy + 20% efficiency + 10% speed + 10% robustness)
    - Seeding: 30% Phase 3-8 winners + 70% random

    Estimated experiments: population_size * n_generations * ~2.5 (quick + full eval)
    """

    # Population parameters
    population_size: int = 50
    n_elite: int = 2
    tournament_size: int = 3

    # Evolution rates
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2

    # Training parameters
    quick_epochs: int = 50
    full_epochs: int = 500
    quick_accuracy_threshold: float = 0.55

    # Architecture constraints
    min_depth: int = 1
    max_depth: int = 10
    min_width: int = 2
    max_width: int = 64

    # Datasets
    datasets: List[str] = field(default_factory=lambda: [
        'xor', 'moons', 'circles', 'spirals',
    ])

    # Number of generations
    n_generations: int = 30

    # Diversity parameters
    min_genome_distance: float = 0.15

    # Early stopping
    early_stop_patience: int = 10
    early_stop_min_improvement: float = 0.001

    # Checkpointing
    checkpoint_every: int = 5

    def get_estimated_experiments(self) -> int:
        """
        Estimate total training runs.

        Each genome is evaluated ~2.5 times on average:
        - Quick screening (all genomes)
        - Full evaluation (~60% pass screening)
        """
        return int(self.population_size * self.n_generations * 2.5 * len(self.datasets))

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            'population_size': self.population_size,
            'n_generations': self.n_generations,
            'n_elite': self.n_elite,
            'tournament_size': self.tournament_size,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'quick_epochs': self.quick_epochs,
            'full_epochs': self.full_epochs,
            'quick_accuracy_threshold': self.quick_accuracy_threshold,
            'architecture_constraints': {
                'min_depth': self.min_depth,
                'max_depth': self.max_depth,
                'min_width': self.min_width,
                'max_width': self.max_width,
            },
            'datasets': self.datasets,
            'estimated_experiments': self.get_estimated_experiments(),
        }

    def to_evolution_config(self) -> 'EvolutionConfig':
        """Convert to EvolutionConfig for the engine."""
        from ..evolution.engine import EvolutionConfig
        return EvolutionConfig(
            population_size=self.population_size,
            n_elite=self.n_elite,
            tournament_size=self.tournament_size,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            quick_epochs=self.quick_epochs,
            full_epochs=self.full_epochs,
            quick_accuracy_threshold=self.quick_accuracy_threshold,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            min_width=self.min_width,
            max_width=self.max_width,
            min_genome_distance=self.min_genome_distance,
            early_stop_patience=self.early_stop_patience,
            early_stop_min_improvement=self.early_stop_min_improvement,
            checkpoint_every=self.checkpoint_every,
        )
