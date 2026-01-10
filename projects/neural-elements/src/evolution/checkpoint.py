"""
Checkpointing for evolutionary runs.

Enables:
- Saving evolution state for resumption
- Recording generation history
- Preserving best individuals (Pareto front / champions)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from .genome import Genome


@dataclass
class EvolutionCheckpoint:
    """
    Checkpoint for resuming evolutionary runs.

    Contains all state needed to continue evolution from a saved point.
    """
    run_id: str
    generation: int
    population: List[Dict[str, Any]]  # Serialized genomes
    champions: List[Dict[str, Any]]   # Best individuals found so far
    history: Dict[str, List[Any]]     # Generation-by-generation stats
    config: Dict[str, Any]            # Evolution configuration
    dataset_name: str                 # Dataset being optimized for
    timestamp: str
    total_evaluations: int = 0
    early_stopped: bool = False
    early_stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionCheckpoint':
        """Create from dictionary."""
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save checkpoint to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'EvolutionCheckpoint':
        """Load checkpoint from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_population(self) -> List[Genome]:
        """Deserialize population to Genome objects."""
        return [Genome.from_dict(g) for g in self.population]

    def get_champions(self) -> List[Genome]:
        """Deserialize champions to Genome objects."""
        return [Genome.from_dict(g) for g in self.champions]


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    min_fitness: float
    std_fitness: float
    best_accuracy: float
    mean_accuracy: float
    population_size: int
    unique_architectures: int
    mean_depth: float
    mean_params: float
    evaluations_this_gen: int
    failed_evaluations: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvolutionHistory:
    """
    Tracks evolution progress over generations.

    Records per-generation statistics for analysis and visualization.
    """

    def __init__(self):
        self.generations: List[GenerationStats] = []
        self.best_genome_per_gen: List[Dict[str, Any]] = []
        self.fitness_trajectory: List[float] = []
        self.diversity_trajectory: List[float] = []

    def record_generation(
        self,
        generation: int,
        population: List[Genome],
        evaluations: int,
        failed: int,
    ) -> GenerationStats:
        """
        Record statistics for a completed generation.

        Args:
            generation: Generation number
            population: Current population with fitness evaluated
            evaluations: Number of fitness evaluations this generation
            failed: Number of failed evaluations

        Returns:
            GenerationStats for this generation
        """
        from .fitness import weighted_fitness
        import numpy as np

        # Get fitness values
        evaluated = [g for g in population if g.fitness is not None]
        if not evaluated:
            fitnesses = [0.0]
            accuracies = [0.0]
        else:
            fitnesses = [weighted_fitness(g.fitness) for g in evaluated]
            accuracies = [g.fitness.test_accuracy for g in evaluated]

        # Get best genome
        if evaluated:
            best_genome = max(evaluated, key=lambda g: weighted_fitness(g.fitness))
            self.best_genome_per_gen.append(best_genome.to_dict())
        else:
            self.best_genome_per_gen.append(None)

        # Count unique architectures
        arch_signatures = set()
        for g in population:
            sig = (tuple(g.hidden_layers), tuple(g.activations), g.skip_connections)
            arch_signatures.add(sig)

        stats = GenerationStats(
            generation=generation,
            best_fitness=max(fitnesses),
            mean_fitness=float(np.mean(fitnesses)),
            min_fitness=min(fitnesses),
            std_fitness=float(np.std(fitnesses)),
            best_accuracy=max(accuracies),
            mean_accuracy=float(np.mean(accuracies)),
            population_size=len(population),
            unique_architectures=len(arch_signatures),
            mean_depth=float(np.mean([g.depth for g in population])),
            mean_params=float(np.mean([g.total_params for g in population])),
            evaluations_this_gen=evaluations,
            failed_evaluations=failed,
            timestamp=datetime.now().isoformat(),
        )

        self.generations.append(stats)
        self.fitness_trajectory.append(stats.best_fitness)
        self.diversity_trajectory.append(
            len(arch_signatures) / len(population) if population else 0
        )

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary for serialization."""
        return {
            'generations': [g.to_dict() for g in self.generations],
            'best_genome_per_gen': self.best_genome_per_gen,
            'fitness_trajectory': self.fitness_trajectory,
            'diversity_trajectory': self.diversity_trajectory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionHistory':
        """Restore history from dictionary."""
        history = cls()
        history.generations = [
            GenerationStats(**g) for g in data.get('generations', [])
        ]
        history.best_genome_per_gen = data.get('best_genome_per_gen', [])
        history.fitness_trajectory = data.get('fitness_trajectory', [])
        history.diversity_trajectory = data.get('diversity_trajectory', [])
        return history

    def get_improvement_rate(self, window: int = 5) -> float:
        """
        Calculate recent improvement rate.

        Used for early stopping detection.

        Args:
            window: Number of recent generations to consider

        Returns:
            Improvement rate (positive = improving)
        """
        if len(self.fitness_trajectory) < window + 1:
            return float('inf')  # Not enough data

        recent = self.fitness_trajectory[-window:]
        older = self.fitness_trajectory[-(window + 1):-1]

        recent_best = max(recent)
        older_best = max(older)

        return recent_best - older_best

    def should_early_stop(
        self,
        patience: int = 10,
        min_improvement: float = 0.001,
    ) -> bool:
        """
        Check if evolution should stop early.

        Args:
            patience: Generations without improvement before stopping
            min_improvement: Minimum improvement to count as progress

        Returns:
            True if should stop, False otherwise
        """
        # Need at least patience + 1 generations to compare
        if len(self.fitness_trajectory) <= patience:
            return False

        recent_best = max(self.fitness_trajectory[-patience:])
        older_best = max(self.fitness_trajectory[:-patience])

        improvement = recent_best - older_best
        return improvement < min_improvement


def generate_run_id() -> str:
    """Generate unique run identifier."""
    import uuid
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_uuid = uuid.uuid4().hex[:6]
    return f"evo_{timestamp}_{short_uuid}"
