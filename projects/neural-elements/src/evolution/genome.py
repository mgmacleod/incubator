"""
Genome representation for evolutionary neural architecture search.

A Genome encodes a neural element architecture as a set of genes that can
be mutated, crossed over, and evaluated for fitness.

Key features:
- Variable-length hidden layers (depth is evolvable)
- Per-layer activation functions (mixed activations supported)
- Skip connections toggle
- Learning rate as a tunable gene
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
import uuid


# Available activation functions (from src/core/activations.py)
AVAILABLE_ACTIVATIONS = [
    'relu',
    'leaky_relu',
    'tanh',
    'sigmoid',
    'gelu',
    'sine',
]

# Width choices for layer mutations
WIDTH_CHOICES = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def generate_genome_id(generation: int = 0, prefix: str = '') -> str:
    """Generate a unique genome identifier."""
    short_uuid = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}_gen{generation}_{short_uuid}"
    return f"gen{generation}_{short_uuid}"


@dataclass
class Genome:
    """
    Genetic representation of a neural element architecture.

    Attributes:
        hidden_layers: List of neuron counts per hidden layer, e.g. [16, 8, 4]
        activations: Per-layer activation function, e.g. ['relu', 'tanh', 'relu']
        skip_connections: Whether to use residual/skip connections
        learning_rate: Training learning rate (log-scale range: 0.001 to 1.0)
        genome_id: Unique identifier for this genome
        generation: Generation number when this genome was created
        parents: Tuple of parent genome IDs (for lineage tracking)
        fitness: Fitness scores after evaluation (None if not yet evaluated)
    """
    hidden_layers: List[int]
    activations: List[str]
    skip_connections: bool
    learning_rate: float
    genome_id: str
    generation: int
    parents: Tuple[str, str]
    fitness: Optional['FitnessScores'] = None  # Populated after evaluation

    def __post_init__(self):
        """Validate genome consistency."""
        # Ensure activations list matches hidden_layers length
        if len(self.activations) != len(self.hidden_layers):
            raise ValueError(
                f"Activations length ({len(self.activations)}) must match "
                f"hidden_layers length ({len(self.hidden_layers)})"
            )
        # Validate activation names
        for act in self.activations:
            if act not in AVAILABLE_ACTIVATIONS:
                raise ValueError(f"Unknown activation: {act}")
        # Validate learning rate range
        if not 0.0001 <= self.learning_rate <= 2.0:
            raise ValueError(f"Learning rate {self.learning_rate} out of range [0.0001, 2.0]")

    @property
    def depth(self) -> int:
        """Number of hidden layers."""
        return len(self.hidden_layers)

    @property
    def total_params(self) -> int:
        """
        Calculate total number of trainable parameters.

        Assumes input_dim=2, output_dim=1 (standard for toy datasets).
        """
        input_dim = 2
        output_dim = 1
        params = 0
        prev_dim = input_dim

        for width in self.hidden_layers:
            params += prev_dim * width  # weights
            params += width  # biases
            prev_dim = width

        # Output layer
        params += prev_dim * output_dim + output_dim
        return params

    @property
    def primary_activation(self) -> str:
        """
        Return the most common activation (for display/naming).

        If all activations are the same, returns that activation.
        Otherwise returns the first activation with a '+' suffix.
        """
        if len(set(self.activations)) == 1:
            return self.activations[0]
        # Return first with indicator of mixed
        return f"{self.activations[0]}+"

    @property
    def architecture_string(self) -> str:
        """Human-readable architecture description."""
        layers_str = '-'.join(str(w) for w in self.hidden_layers)
        act_str = self.primary_activation
        skip_str = '+skip' if self.skip_connections else ''
        return f"{act_str}[{layers_str}]{skip_str}"

    def to_element_config(self) -> Dict[str, Any]:
        """
        Convert genome to NeuralElement-compatible configuration.

        Note: NeuralElement currently only supports a single activation for
        all layers. When activations are mixed, we use the first activation
        and rely on the custom forward pass in the evolution worker to
        handle per-layer activations.
        """
        return {
            'hidden_layers': self.hidden_layers.copy(),
            'activation': self.activations[0],  # Primary activation
            'skip_connections': self.skip_connections,
            'input_dim': 2,
            'output_dim': 1,
            'bias': True,
        }

    def has_mixed_activations(self) -> bool:
        """Check if this genome uses different activations per layer."""
        return len(set(self.activations)) > 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Note: fitness is converted separately since it's a dataclass.
        """
        d = {
            'hidden_layers': self.hidden_layers,
            'activations': self.activations,
            'skip_connections': self.skip_connections,
            'learning_rate': self.learning_rate,
            'genome_id': self.genome_id,
            'generation': self.generation,
            'parents': list(self.parents),
        }
        if self.fitness is not None:
            d['fitness'] = self.fitness.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
        """Create Genome from dictionary (e.g., loaded from JSON)."""
        # Import here to avoid circular dependency
        from .fitness import FitnessScores

        fitness = None
        if 'fitness' in data and data['fitness'] is not None:
            fitness = FitnessScores.from_dict(data['fitness'])

        return cls(
            hidden_layers=data['hidden_layers'],
            activations=data['activations'],
            skip_connections=data['skip_connections'],
            learning_rate=data['learning_rate'],
            genome_id=data['genome_id'],
            generation=data['generation'],
            parents=tuple(data['parents']),
            fitness=fitness,
        )

    def copy(self) -> 'Genome':
        """Create a deep copy of this genome."""
        return Genome(
            hidden_layers=self.hidden_layers.copy(),
            activations=self.activations.copy(),
            skip_connections=self.skip_connections,
            learning_rate=self.learning_rate,
            genome_id=self.genome_id,
            generation=self.generation,
            parents=self.parents,
            fitness=self.fitness,
        )

    def __repr__(self) -> str:
        fitness_str = f", fitness={self.fitness.test_accuracy:.3f}" if self.fitness else ""
        return (
            f"Genome(id={self.genome_id}, arch={self.architecture_string}, "
            f"params={self.total_params}, gen={self.generation}{fitness_str})"
        )


def create_random_genome(
    generation: int = 0,
    min_depth: int = 1,
    max_depth: int = 6,
    min_width: int = 2,
    max_width: int = 64,
    prefix: str = 'rand',
) -> Genome:
    """
    Create a random genome with constraints.

    Args:
        generation: Generation number for this genome
        min_depth: Minimum number of hidden layers
        max_depth: Maximum number of hidden layers
        min_width: Minimum layer width
        max_width: Maximum layer width
        prefix: Prefix for genome ID

    Returns:
        A randomly initialized Genome
    """
    import random

    # Random depth
    depth = random.randint(min_depth, max_depth)

    # Random widths (from WIDTH_CHOICES within bounds)
    valid_widths = [w for w in WIDTH_CHOICES if min_width <= w <= max_width]
    hidden_layers = [random.choice(valid_widths) for _ in range(depth)]

    # Random activations per layer
    activations = [random.choice(AVAILABLE_ACTIVATIONS) for _ in range(depth)]

    # Random skip connections
    skip_connections = random.choice([True, False])

    # Random learning rate (log-uniform between 0.01 and 1.0)
    import numpy as np
    learning_rate = float(10 ** np.random.uniform(-2, 0))

    return Genome(
        hidden_layers=hidden_layers,
        activations=activations,
        skip_connections=skip_connections,
        learning_rate=learning_rate,
        genome_id=generate_genome_id(generation, prefix),
        generation=generation,
        parents=('random', 'random'),
    )


def genome_from_element_config(
    config: Dict[str, Any],
    generation: int = 0,
    prefix: str = 'seed',
) -> Genome:
    """
    Create a Genome from an existing element configuration.

    Useful for seeding the initial population with known good architectures.

    Args:
        config: Element configuration dict with 'hidden_layers' and 'activation'
        generation: Generation number
        prefix: Prefix for genome ID

    Returns:
        A Genome representing the element configuration
    """
    hidden_layers = config['hidden_layers']
    activation = config.get('activation', 'relu')

    return Genome(
        hidden_layers=hidden_layers.copy(),
        activations=[activation] * len(hidden_layers),
        skip_connections=config.get('skip_connections', False),
        learning_rate=config.get('learning_rate', 0.1),
        genome_id=generate_genome_id(generation, prefix),
        generation=generation,
        parents=('seed', 'seed'),
    )
