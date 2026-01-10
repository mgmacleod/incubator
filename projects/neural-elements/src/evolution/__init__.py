"""
Neural Elements - Phase 9: Evolutionary Discovery

This module provides a genetic algorithm framework for discovering optimal
neural element architectures through evolutionary search.

Key components:
- Genome: Representation of a neural network architecture
- FitnessScores: Multi-objective fitness evaluation
- EvolutionEngine: Main evolutionary optimization loop
- Operators: Selection, crossover, and mutation operations

Example usage:
    from src.evolution import EvolutionEngine, EvolutionConfig
    from src.datasets.toy import get_dataset

    # Get dataset
    X, y = get_dataset('spirals', n_samples=200)
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    # Configure evolution
    config = EvolutionConfig(population_size=50, n_workers=8)

    # Create and run engine
    engine = EvolutionEngine(config, X_train, y_train, X_test, y_test)
    engine.initialize_population()
    result = engine.evolve(n_generations=30)

    print(f"Best accuracy: {result.best_accuracy:.3f}")
"""

from .genome import Genome, create_random_genome, genome_from_element_config
from .fitness import FitnessScores, compute_fitness, weighted_fitness
from .operators import (
    tournament_selection,
    elitism_selection,
    layer_crossover,
    uniform_crossover,
    mutate_genome,
)
from .population import (
    create_initial_population,
    compute_genome_distance,
    enforce_diversity,
    get_phase_winners,
)
from .engine import EvolutionEngine, EvolutionConfig, EvolutionResult
from .checkpoint import EvolutionCheckpoint, EvolutionHistory

__all__ = [
    # Core classes
    'Genome',
    'FitnessScores',
    'EvolutionEngine',
    'EvolutionConfig',
    'EvolutionResult',
    'EvolutionCheckpoint',
    'EvolutionHistory',
    # Genome helpers
    'create_random_genome',
    'genome_from_element_config',
    # Fitness
    'compute_fitness',
    'weighted_fitness',
    # Operators
    'tournament_selection',
    'elitism_selection',
    'layer_crossover',
    'uniform_crossover',
    'mutate_genome',
    # Population
    'create_initial_population',
    'compute_genome_distance',
    'enforce_diversity',
    'get_phase_winners',
]
