"""
Tests for Phase 9: Evolutionary Discovery module.

Run with: python -m pytest tests/test_evolution.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evolution.genome import (
    Genome,
    create_random_genome,
    genome_from_element_config,
    generate_genome_id,
    AVAILABLE_ACTIVATIONS,
    WIDTH_CHOICES,
)
from src.evolution.fitness import (
    FitnessScores,
    weighted_fitness,
    compute_fitness,
)
from src.evolution.operators import (
    tournament_selection,
    elitism_selection,
    layer_crossover,
    uniform_crossover,
    mutate_genome,
)
from src.evolution.population import (
    create_initial_population,
    compute_genome_distance,
    enforce_diversity,
    get_population_stats,
)
from src.evolution.checkpoint import (
    EvolutionCheckpoint,
    EvolutionHistory,
    GenerationStats,
)


class TestGenome:
    """Tests for Genome class."""

    def test_genome_creation(self):
        """Test basic genome creation."""
        genome = Genome(
            hidden_layers=[8, 4, 8],
            activations=['relu', 'tanh', 'relu'],
            skip_connections=False,
            learning_rate=0.1,
            genome_id='test_001',
            generation=0,
            parents=('a', 'b'),
        )

        assert genome.depth == 3
        assert genome.hidden_layers == [8, 4, 8]
        assert genome.activations == ['relu', 'tanh', 'relu']
        assert genome.learning_rate == 0.1

    def test_genome_total_params(self):
        """Test parameter counting."""
        # Simple 1-layer network: 2->8->1
        # Weights: 2*8 + 8*1 = 24
        # Biases: 8 + 1 = 9
        # Total: 33
        genome = Genome(
            hidden_layers=[8],
            activations=['relu'],
            skip_connections=False,
            learning_rate=0.1,
            genome_id='test',
            generation=0,
            parents=('a', 'b'),
        )
        assert genome.total_params == 33

    def test_genome_serialization(self):
        """Test genome to/from dict."""
        genome = Genome(
            hidden_layers=[16, 8],
            activations=['relu', 'tanh'],
            skip_connections=True,
            learning_rate=0.05,
            genome_id='test_002',
            generation=5,
            parents=('p1', 'p2'),
        )

        # Serialize
        d = genome.to_dict()
        assert d['hidden_layers'] == [16, 8]
        assert d['activations'] == ['relu', 'tanh']
        assert d['skip_connections'] is True

        # Deserialize
        restored = Genome.from_dict(d)
        assert restored.hidden_layers == genome.hidden_layers
        assert restored.activations == genome.activations
        assert restored.genome_id == genome.genome_id

    def test_create_random_genome(self):
        """Test random genome creation."""
        np.random.seed(42)
        genome = create_random_genome(
            generation=1,
            min_depth=2,
            max_depth=5,
        )

        assert 2 <= genome.depth <= 5
        assert all(a in AVAILABLE_ACTIVATIONS for a in genome.activations)
        assert genome.generation == 1

    def test_genome_from_element_config(self):
        """Test creating genome from element config."""
        config = {
            'hidden_layers': [8, 8, 8],
            'activation': 'sine',
            'skip_connections': True,
        }

        genome = genome_from_element_config(config, generation=0)

        assert genome.hidden_layers == [8, 8, 8]
        assert genome.activations == ['sine', 'sine', 'sine']
        assert genome.skip_connections is True

    def test_genome_validation(self):
        """Test genome validation."""
        # Mismatched lengths should raise
        with pytest.raises(ValueError):
            Genome(
                hidden_layers=[8, 8],
                activations=['relu'],  # Wrong length
                skip_connections=False,
                learning_rate=0.1,
                genome_id='test',
                generation=0,
                parents=('a', 'b'),
            )

        # Invalid activation should raise
        with pytest.raises(ValueError):
            Genome(
                hidden_layers=[8],
                activations=['invalid_activation'],
                skip_connections=False,
                learning_rate=0.1,
                genome_id='test',
                generation=0,
                parents=('a', 'b'),
            )


class TestFitness:
    """Tests for fitness evaluation."""

    def test_fitness_scores_creation(self):
        """Test FitnessScores creation."""
        scores = FitnessScores(
            test_accuracy=0.85,
            parameter_efficiency=0.7,
            convergence_speed=0.6,
            noise_robustness=0.8,
            training_failed=False,
        )

        assert scores.test_accuracy == 0.85
        assert not scores.training_failed

    def test_weighted_fitness(self):
        """Test weighted fitness calculation."""
        scores = FitnessScores(
            test_accuracy=1.0,
            parameter_efficiency=1.0,
            convergence_speed=1.0,
            noise_robustness=1.0,
            training_failed=False,
        )

        # Perfect scores should give 1.0
        assert weighted_fitness(scores) == pytest.approx(1.0)

        # Failed training should give 0
        failed_scores = FitnessScores(
            test_accuracy=0.9,
            parameter_efficiency=0.8,
            convergence_speed=0.7,
            noise_robustness=0.9,
            training_failed=True,
        )
        assert weighted_fitness(failed_scores) == 0.0

    def test_fitness_serialization(self):
        """Test FitnessScores to/from dict."""
        scores = FitnessScores(
            test_accuracy=0.9,
            parameter_efficiency=0.75,
            convergence_speed=0.5,
            noise_robustness=0.85,
            training_failed=False,
            train_accuracy=0.92,
        )

        d = scores.to_dict()
        restored = FitnessScores.from_dict(d)

        assert restored.test_accuracy == scores.test_accuracy
        assert restored.train_accuracy == scores.train_accuracy


class TestOperators:
    """Tests for evolutionary operators."""

    @pytest.fixture
    def sample_population(self):
        """Create sample population for testing."""
        population = []
        for i in range(10):
            genome = create_random_genome(generation=0)
            genome.fitness = FitnessScores(
                test_accuracy=0.5 + i * 0.05,  # 0.5 to 0.95
                parameter_efficiency=0.5,
                convergence_speed=0.5,
                noise_robustness=0.5,
                training_failed=False,
            )
            population.append(genome)
        return population

    def test_tournament_selection(self, sample_population):
        """Test tournament selection."""
        selected = tournament_selection(sample_population, n_select=5, tournament_size=3)

        assert len(selected) == 5
        # Should tend toward fitter individuals
        avg_fitness = np.mean([weighted_fitness(g.fitness) for g in selected])
        assert avg_fitness > 0.5  # Better than random

    def test_elitism_selection(self, sample_population):
        """Test elitism selection."""
        elite = elitism_selection(sample_population, n_elite=2)

        assert len(elite) == 2
        # Should be the top 2 by fitness
        fitnesses = [weighted_fitness(g.fitness) for g in elite]
        # Top 2 have accuracies of 0.95 and 0.90, weighted gives ~0.72 and ~0.69
        assert all(f >= 0.65 for f in fitnesses)  # Top performers

    def test_layer_crossover(self):
        """Test layer crossover."""
        parent1 = Genome(
            hidden_layers=[16, 8, 4],
            activations=['relu', 'relu', 'relu'],
            skip_connections=False,
            learning_rate=0.1,
            genome_id='p1',
            generation=0,
            parents=('a', 'b'),
        )
        parent2 = Genome(
            hidden_layers=[8, 16, 8, 8],
            activations=['tanh', 'tanh', 'tanh', 'tanh'],
            skip_connections=True,
            learning_rate=0.05,
            genome_id='p2',
            generation=0,
            parents=('c', 'd'),
        )

        child1, child2 = layer_crossover(parent1, parent2, generation=1)

        # Children should be valid genomes
        assert len(child1.hidden_layers) == len(child1.activations)
        assert len(child2.hidden_layers) == len(child2.activations)
        assert child1.generation == 1
        assert child1.parents == ('p1', 'p2')

    def test_uniform_crossover(self):
        """Test uniform crossover."""
        parent1 = Genome(
            hidden_layers=[8, 8, 8],
            activations=['relu', 'relu', 'relu'],
            skip_connections=False,
            learning_rate=0.1,
            genome_id='p1',
            generation=0,
            parents=('a', 'b'),
        )
        parent2 = Genome(
            hidden_layers=[16, 16, 16],
            activations=['tanh', 'tanh', 'tanh'],
            skip_connections=True,
            learning_rate=0.05,
            genome_id='p2',
            generation=0,
            parents=('c', 'd'),
        )

        np.random.seed(42)
        child1, child2 = uniform_crossover(parent1, parent2, generation=1)

        # Children should have some genes from each parent
        assert len(child1.hidden_layers) >= 3
        assert child1.generation == 1

    def test_mutation(self):
        """Test genome mutation."""
        genome = Genome(
            hidden_layers=[8, 8, 8],
            activations=['relu', 'relu', 'relu'],
            skip_connections=False,
            learning_rate=0.1,
            genome_id='orig',
            generation=0,
            parents=('a', 'b'),
        )

        np.random.seed(42)
        mutated = mutate_genome(genome, generation=1, mutation_rate=0.5)

        # Should be a different genome
        assert mutated.genome_id != genome.genome_id
        assert mutated.generation == 1
        assert mutated.parents == ('orig', 'mutation')

        # Should still be valid
        assert len(mutated.hidden_layers) == len(mutated.activations)
        assert all(a in AVAILABLE_ACTIVATIONS for a in mutated.activations)


class TestPopulation:
    """Tests for population management."""

    def test_create_initial_population(self):
        """Test initial population creation."""
        phase_winners = [
            {'hidden_layers': [8, 8], 'activation': 'relu'},
            {'hidden_layers': [8, 8, 8], 'activation': 'tanh'},
        ]

        population = create_initial_population(
            population_size=10,
            phase_winners=phase_winners,
            seed=42,
            seed_ratio=0.3,
        )

        assert len(population) == 10
        # Should have some seeded individuals
        seeded = [g for g in population if 'seed' in g.genome_id]
        assert len(seeded) >= 2

    def test_genome_distance(self):
        """Test genome distance calculation."""
        g1 = Genome(
            hidden_layers=[8, 8, 8],
            activations=['relu', 'relu', 'relu'],
            skip_connections=False,
            learning_rate=0.1,
            genome_id='g1',
            generation=0,
            parents=('a', 'b'),
        )
        g2 = Genome(
            hidden_layers=[8, 8, 8],
            activations=['relu', 'relu', 'relu'],
            skip_connections=False,
            learning_rate=0.1,
            genome_id='g2',
            generation=0,
            parents=('c', 'd'),
        )
        g3 = Genome(
            hidden_layers=[16, 4],
            activations=['tanh', 'tanh'],
            skip_connections=True,
            learning_rate=0.01,
            genome_id='g3',
            generation=0,
            parents=('e', 'f'),
        )

        # Identical architectures should have distance 0
        assert compute_genome_distance(g1, g2) == pytest.approx(0.0, abs=0.01)

        # Different architectures should have distance > 0
        assert compute_genome_distance(g1, g3) > 0.2

    def test_enforce_diversity(self):
        """Test diversity enforcement."""
        # Create population with duplicates
        population = []
        for i in range(10):
            genome = Genome(
                hidden_layers=[8, 8],
                activations=['relu', 'relu'],
                skip_connections=False,
                learning_rate=0.1,
                genome_id=f'g{i}',
                generation=0,
                parents=('a', 'b'),
            )
            genome.fitness = FitnessScores(
                test_accuracy=0.9 - i * 0.01,
                parameter_efficiency=0.5,
                convergence_speed=0.5,
                noise_robustness=0.5,
                training_failed=False,
            )
            population.append(genome)

        # Should remove most duplicates (all same architecture)
        diverse = enforce_diversity(population, min_distance=0.1)
        assert len(diverse) < len(population)

    def test_population_stats(self):
        """Test population statistics."""
        population = create_initial_population(population_size=20, seed=42)

        stats = get_population_stats(population)

        assert stats['size'] == 20
        assert 'depth_range' in stats
        assert 'activation_counts' in stats


class TestCheckpoint:
    """Tests for checkpointing."""

    def test_evolution_checkpoint(self, tmp_path):
        """Test checkpoint save/load."""
        checkpoint = EvolutionCheckpoint(
            run_id='test_run',
            generation=5,
            population=[
                create_random_genome(generation=5).to_dict()
                for _ in range(3)
            ],
            champions=[],
            history={'fitness': [0.5, 0.6, 0.7]},
            config={'population_size': 50},
            dataset_name='spirals',
            timestamp='2024-01-01T00:00:00',
            total_evaluations=100,
        )

        # Save
        checkpoint_path = tmp_path / 'checkpoint.json'
        checkpoint.save(checkpoint_path)

        # Load
        loaded = EvolutionCheckpoint.load(checkpoint_path)

        assert loaded.run_id == 'test_run'
        assert loaded.generation == 5
        assert len(loaded.population) == 3
        assert loaded.total_evaluations == 100

    def test_evolution_history(self):
        """Test evolution history tracking."""
        history = EvolutionHistory()

        # Create sample population
        population = []
        for i in range(10):
            genome = create_random_genome(generation=1)
            genome.fitness = FitnessScores(
                test_accuracy=0.5 + i * 0.05,
                parameter_efficiency=0.5,
                convergence_speed=0.5,
                noise_robustness=0.5,
                training_failed=False,
            )
            population.append(genome)

        # Record generation
        stats = history.record_generation(
            generation=1,
            population=population,
            evaluations=20,
            failed=2,
        )

        assert stats.generation == 1
        assert stats.best_accuracy == pytest.approx(0.95)
        assert stats.evaluations_this_gen == 20
        assert len(history.fitness_trajectory) == 1

    def test_early_stopping_detection(self):
        """Test early stopping logic."""
        history = EvolutionHistory()

        # Simulate improving generations
        for gen in range(5):
            population = [create_random_genome(generation=gen)]
            population[0].fitness = FitnessScores(
                test_accuracy=0.5 + gen * 0.1,  # Improving
                parameter_efficiency=0.5,
                convergence_speed=0.5,
                noise_robustness=0.5,
                training_failed=False,
            )
            history.record_generation(gen, population, 10, 0)

        # Should not stop (improving)
        assert not history.should_early_stop(patience=3, min_improvement=0.01)

        # Simulate stagnation
        for gen in range(5, 15):
            population = [create_random_genome(generation=gen)]
            population[0].fitness = FitnessScores(
                test_accuracy=0.9,  # Stagnant
                parameter_efficiency=0.5,
                convergence_speed=0.5,
                noise_robustness=0.5,
                training_failed=False,
            )
            history.record_generation(gen, population, 10, 0)

        # Should stop (no improvement)
        assert history.should_early_stop(patience=5, min_improvement=0.01)


class TestIntegration:
    """Integration tests."""

    def test_full_genome_lifecycle(self):
        """Test complete genome lifecycle."""
        # Create
        genome = create_random_genome(generation=0)

        # Serialize
        d = genome.to_dict()

        # Deserialize
        restored = Genome.from_dict(d)

        # Convert to element config
        config = restored.to_element_config()

        assert 'hidden_layers' in config
        assert 'activation' in config

    def test_population_evolution_step(self):
        """Test one step of population evolution."""
        # Create initial population
        population = create_initial_population(population_size=10, seed=42)

        # Assign dummy fitness
        for i, genome in enumerate(population):
            genome.fitness = FitnessScores(
                test_accuracy=0.5 + np.random.random() * 0.5,
                parameter_efficiency=0.5,
                convergence_speed=0.5,
                noise_robustness=0.5,
                training_failed=False,
            )

        # Selection
        elite = elitism_selection(population, n_elite=2)
        parents = tournament_selection(population, n_select=8)

        # Crossover
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            c1, c2 = layer_crossover(parents[i], parents[i+1], generation=1)
            offspring.extend([c1, c2])

        # Mutation
        mutated = [mutate_genome(g, generation=1) for g in offspring]

        # New population
        new_population = elite + mutated

        assert len(new_population) == 10
        assert all(g.generation <= 1 for g in new_population)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
