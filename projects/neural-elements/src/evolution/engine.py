"""
Main evolutionary optimization engine.

Orchestrates the evolution loop:
1. Initialize population
2. Evaluate fitness (parallel)
3. Select parents
4. Create offspring via crossover/mutation
5. Manage diversity
6. Checkpoint progress
7. Repeat until convergence
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time
import random
import numpy as np

from .genome import Genome, create_random_genome
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
    get_phase_winners,
    enforce_diversity,
    get_population_stats,
)
from .checkpoint import (
    EvolutionCheckpoint,
    EvolutionHistory,
    generate_run_id,
)


@dataclass
class EvolutionResult:
    """Results from an evolution run."""
    run_id: str
    generations_completed: int
    total_evaluations: int
    best_fitness: float
    best_accuracy: float
    champions: List[Genome]  # Top performers
    history: EvolutionHistory
    final_population: List[Genome]
    runtime_seconds: float
    early_stopped: bool
    early_stop_reason: Optional[str] = None

    def get_best_genome(self) -> Optional[Genome]:
        """Return the single best genome found."""
        if not self.champions:
            return None
        return max(self.champions, key=lambda g: weighted_fitness(g.fitness) if g.fitness else 0)

    def summary(self) -> str:
        """Generate summary string."""
        best = self.get_best_genome()
        lines = [
            f"Evolution Run: {self.run_id}",
            f"Generations: {self.generations_completed}",
            f"Total evaluations: {self.total_evaluations}",
            f"Best fitness: {self.best_fitness:.4f}",
            f"Best accuracy: {self.best_accuracy:.4f}",
            f"Runtime: {self.runtime_seconds:.1f}s",
            f"Early stopped: {self.early_stopped}",
        ]
        if best:
            lines.append(f"Best architecture: {best.architecture_string}")
        return '\n'.join(lines)


@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""
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

    # Diversity parameters
    min_genome_distance: float = 0.15

    # Early stopping
    early_stop_patience: int = 10
    early_stop_min_improvement: float = 0.001

    # Checkpointing
    checkpoint_every: int = 5
    checkpoint_dir: Optional[str] = None

    # Parallelization
    n_workers: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'population_size': self.population_size,
            'n_elite': self.n_elite,
            'tournament_size': self.tournament_size,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'quick_epochs': self.quick_epochs,
            'full_epochs': self.full_epochs,
            'quick_accuracy_threshold': self.quick_accuracy_threshold,
            'min_depth': self.min_depth,
            'max_depth': self.max_depth,
            'min_width': self.min_width,
            'max_width': self.max_width,
            'min_genome_distance': self.min_genome_distance,
            'early_stop_patience': self.early_stop_patience,
            'early_stop_min_improvement': self.early_stop_min_improvement,
            'checkpoint_every': self.checkpoint_every,
            'n_workers': self.n_workers,
        }


class EvolutionEngine:
    """
    Main evolutionary optimization engine.

    Evolves neural element architectures to optimize fitness on a given dataset.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str = 'unknown',
        run_id: Optional[str] = None,
    ):
        """
        Initialize evolution engine.

        Args:
            config: Evolution configuration
            X_train, y_train: Training data
            X_test, y_test: Test data for fitness evaluation
            dataset_name: Name of dataset (for checkpointing)
            run_id: Optional run identifier (auto-generated if not provided)
        """
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset_name = dataset_name
        self.run_id = run_id or generate_run_id()

        self.population: List[Genome] = []
        self.champions: List[Genome] = []
        self.history = EvolutionHistory()
        self.generation = 0
        self.total_evaluations = 0

        self.n_workers = config.n_workers or max(1, cpu_count() - 1)

        # Checkpoint directory
        if config.checkpoint_dir:
            self.checkpoint_dir = Path(config.checkpoint_dir)
        else:
            self.checkpoint_dir = Path('data/evolution')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def initialize_population(
        self,
        phase_winners: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Create initial population.

        Args:
            phase_winners: Optional list of seed configurations
            seed: Random seed for reproducibility
        """
        if phase_winners is None:
            phase_winners = get_phase_winners()

        self.population = create_initial_population(
            population_size=self.config.population_size,
            phase_winners=phase_winners,
            seed=seed,
            min_depth=self.config.min_depth,
            max_depth=self.config.max_depth,
            min_width=self.config.min_width,
            max_width=self.config.max_width,
        )

        self.generation = 0
        self.total_evaluations = 0
        self.champions = []
        self.history = EvolutionHistory()

    def evaluate_population(
        self,
        use_quick_screen: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Evaluate fitness for all unevaluated genomes.

        Args:
            use_quick_screen: If True, use quick screening to cull bad genomes
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            Number of evaluations performed
        """
        # Find unevaluated genomes
        to_evaluate = [g for g in self.population if g.fitness is None]

        if not to_evaluate:
            return 0

        evaluations = 0
        failed = 0

        # Quick screening pass
        if use_quick_screen:
            quick_results = self._parallel_evaluate(
                to_evaluate,
                epochs=self.config.quick_epochs,
            )

            passing = []
            for genome, scores in quick_results:
                evaluations += 1
                if scores.training_failed:
                    failed += 1
                    genome.fitness = scores
                elif scores.test_accuracy >= self.config.quick_accuracy_threshold:
                    passing.append(genome)
                else:
                    # Failed screening - keep score but mark as evaluated
                    genome.fitness = scores

                if progress_callback:
                    progress_callback(evaluations, len(to_evaluate) * 2)

            # Full evaluation for passing genomes
            if passing:
                full_results = self._parallel_evaluate(
                    passing,
                    epochs=self.config.full_epochs,
                )
                for genome, scores in full_results:
                    evaluations += 1
                    genome.fitness = scores
                    if scores.training_failed:
                        failed += 1

                    if progress_callback:
                        progress_callback(
                            len(to_evaluate) + evaluations - len(passing),
                            len(to_evaluate) * 2
                        )
        else:
            # Direct full evaluation
            results = self._parallel_evaluate(
                to_evaluate,
                epochs=self.config.full_epochs,
            )
            for genome, scores in results:
                evaluations += 1
                genome.fitness = scores
                if scores.training_failed:
                    failed += 1

                if progress_callback:
                    progress_callback(evaluations, len(to_evaluate))

        self.total_evaluations += evaluations
        return evaluations

    def _parallel_evaluate(
        self,
        genomes: List[Genome],
        epochs: int,
    ) -> List[tuple]:
        """
        Evaluate genomes in parallel.

        Args:
            genomes: List of genomes to evaluate
            epochs: Number of training epochs

        Returns:
            List of (genome, FitnessScores) tuples
        """
        # Prepare arguments for workers
        args_list = [
            (g.to_dict(), self.X_train, self.y_train,
             self.X_test, self.y_test, epochs)
            for g in genomes
        ]

        # Run in parallel
        with Pool(self.n_workers) as pool:
            results = pool.map(_evaluate_worker, args_list)

        # Match results back to genomes
        return [(genomes[i], FitnessScores.from_dict(r)) for i, r in enumerate(results)]

    def run_generation(self) -> None:
        """Execute one generation of evolution."""
        self.generation += 1

        # 1. Evaluate fitness
        evaluations = self.evaluate_population(use_quick_screen=True)
        failed = sum(1 for g in self.population if g.fitness and g.fitness.training_failed)

        # 2. Record statistics
        stats = self.history.record_generation(
            generation=self.generation,
            population=self.population,
            evaluations=evaluations,
            failed=failed,
        )

        # 3. Update champions
        self._update_champions()

        # 4. Selection
        elite = elitism_selection(self.population, self.config.n_elite)

        n_offspring = self.config.population_size - len(elite)
        parents = tournament_selection(
            self.population,
            n_select=n_offspring,
            tournament_size=self.config.tournament_size,
        )

        # 5. Crossover
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            if random.random() < self.config.crossover_rate:
                # Randomly choose crossover type
                if random.random() < 0.5:
                    child1, child2 = layer_crossover(
                        parents[i], parents[i + 1], self.generation
                    )
                else:
                    child1, child2 = uniform_crossover(
                        parents[i], parents[i + 1], self.generation
                    )
                offspring.extend([child1, child2])
            else:
                # No crossover - just copy parents
                offspring.append(parents[i].copy())
                offspring.append(parents[i + 1].copy())

        # Handle odd number of parents
        if len(parents) % 2 == 1:
            offspring.append(parents[-1].copy())

        # 6. Mutation
        mutated_offspring = []
        for genome in offspring:
            if random.random() < self.config.mutation_rate:
                mutated = mutate_genome(
                    genome,
                    self.generation,
                    mutation_rate=self.config.mutation_rate,
                    min_depth=self.config.min_depth,
                    max_depth=self.config.max_depth,
                    min_width=self.config.min_width,
                    max_width=self.config.max_width,
                )
                mutated_offspring.append(mutated)
            else:
                mutated_offspring.append(genome)

        # 7. Form new population
        self.population = elite + mutated_offspring[:self.config.population_size - len(elite)]

        # 8. Enforce diversity
        self.population = enforce_diversity(
            self.population,
            min_distance=self.config.min_genome_distance,
        )

        # 9. Refill if needed (diversity removal reduced population)
        while len(self.population) < self.config.population_size:
            random_genome = create_random_genome(
                generation=self.generation,
                min_depth=self.config.min_depth,
                max_depth=self.config.max_depth,
                min_width=self.config.min_width,
                max_width=self.config.max_width,
            )
            self.population.append(random_genome)

    def _update_champions(self, n_champions: int = 10) -> None:
        """Update the list of best genomes found so far."""
        # Get evaluated genomes from current population
        evaluated = [g for g in self.population if g.fitness and not g.fitness.training_failed]

        # Add to champions pool
        all_candidates = self.champions + evaluated

        # Sort by fitness
        sorted_candidates = sorted(
            all_candidates,
            key=lambda g: weighted_fitness(g.fitness) if g.fitness else 0,
            reverse=True,
        )

        # Keep top n_champions (deduplicated by architecture)
        seen_archs = set()
        new_champions = []
        for g in sorted_candidates:
            arch_sig = (tuple(g.hidden_layers), tuple(g.activations), g.skip_connections)
            if arch_sig not in seen_archs:
                seen_archs.add(arch_sig)
                new_champions.append(g.copy())
                if len(new_champions) >= n_champions:
                    break

        self.champions = new_champions

    def evolve(
        self,
        n_generations: int,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
    ) -> EvolutionResult:
        """
        Run full evolutionary optimization.

        Args:
            n_generations: Maximum number of generations
            progress_callback: Optional callback(gen, total_gens, stats)

        Returns:
            EvolutionResult with final population and statistics
        """
        start_time = time.time()
        early_stopped = False
        early_stop_reason = None

        for gen in range(n_generations):
            self.run_generation()

            # Progress callback
            if progress_callback:
                stats = {
                    'generation': self.generation,
                    'best_fitness': self.history.fitness_trajectory[-1],
                    'mean_fitness': self.history.generations[-1].mean_fitness,
                    'evaluations': self.total_evaluations,
                }
                progress_callback(self.generation, n_generations, stats)

            # Checkpoint
            if self.generation % self.config.checkpoint_every == 0:
                self.save_checkpoint()

            # Early stopping check
            if self.history.should_early_stop(
                patience=self.config.early_stop_patience,
                min_improvement=self.config.early_stop_min_improvement,
            ):
                early_stopped = True
                early_stop_reason = (
                    f"No improvement > {self.config.early_stop_min_improvement} "
                    f"in {self.config.early_stop_patience} generations"
                )
                break

        runtime = time.time() - start_time

        # Final checkpoint
        self.save_checkpoint()

        # Get best fitness and accuracy
        best_fitness = max(self.history.fitness_trajectory) if self.history.fitness_trajectory else 0
        best_accuracy = 0
        if self.champions:
            best_accuracy = max(
                g.fitness.test_accuracy for g in self.champions if g.fitness
            )

        return EvolutionResult(
            run_id=self.run_id,
            generations_completed=self.generation,
            total_evaluations=self.total_evaluations,
            best_fitness=best_fitness,
            best_accuracy=best_accuracy,
            champions=self.champions,
            history=self.history,
            final_population=self.population,
            runtime_seconds=runtime,
            early_stopped=early_stopped,
            early_stop_reason=early_stop_reason,
        )

    def save_checkpoint(self) -> Path:
        """Save current evolution state to checkpoint file."""
        checkpoint = EvolutionCheckpoint(
            run_id=self.run_id,
            generation=self.generation,
            population=[g.to_dict() for g in self.population],
            champions=[g.to_dict() for g in self.champions],
            history=self.history.to_dict(),
            config=self.config.to_dict(),
            dataset_name=self.dataset_name,
            timestamp=datetime.now().isoformat(),
            total_evaluations=self.total_evaluations,
        )

        checkpoint_path = self.checkpoint_dir / f"{self.run_id}_gen{self.generation:03d}.json"
        checkpoint.save(checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume evolution from a checkpoint."""
        checkpoint = EvolutionCheckpoint.load(checkpoint_path)

        self.run_id = checkpoint.run_id
        self.generation = checkpoint.generation
        self.population = checkpoint.get_population()
        self.champions = checkpoint.get_champions()
        self.history = EvolutionHistory.from_dict(checkpoint.history)
        self.total_evaluations = checkpoint.total_evaluations
        self.dataset_name = checkpoint.dataset_name


def _evaluate_worker(args: tuple) -> Dict[str, Any]:
    """
    Worker function for parallel fitness evaluation.

    This is a module-level function to enable pickling for multiprocessing.
    """
    genome_dict, X_train, y_train, X_test, y_test, epochs = args

    # Reconstruct genome
    genome = Genome.from_dict(genome_dict)

    # Evaluate fitness
    scores = compute_fitness(
        genome,
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
    )

    return scores.to_dict()
