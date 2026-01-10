"""
Evolutionary operators: selection, crossover, and mutation.

These operators drive the evolutionary search by:
- Selecting fit individuals for reproduction
- Combining parent genomes through crossover
- Introducing variation through mutation
"""

import random
from typing import List, Tuple, Optional
import numpy as np

from .genome import (
    Genome,
    generate_genome_id,
    AVAILABLE_ACTIVATIONS,
    WIDTH_CHOICES,
)
from .fitness import weighted_fitness


# =============================================================================
# Selection Operators
# =============================================================================

def tournament_selection(
    population: List[Genome],
    n_select: int,
    tournament_size: int = 3,
) -> List[Genome]:
    """
    Tournament selection with replacement.

    Randomly selects tournament_size individuals and keeps the best.
    Repeat n_select times.

    Args:
        population: Current population with fitness scores
        n_select: Number of individuals to select
        tournament_size: Number of individuals per tournament

    Returns:
        List of selected genomes (may contain duplicates)
    """
    selected = []

    for _ in range(n_select):
        # Random tournament
        tournament_size_actual = min(tournament_size, len(population))
        contestants = random.sample(population, tournament_size_actual)

        # Select winner by weighted fitness
        winner = max(
            contestants,
            key=lambda g: weighted_fitness(g.fitness) if g.fitness else 0.0
        )
        selected.append(winner)

    return selected


def elitism_selection(
    population: List[Genome],
    n_elite: int = 2,
) -> List[Genome]:
    """
    Preserve the top n_elite individuals unchanged.

    Args:
        population: Current population with fitness scores
        n_elite: Number of elite individuals to preserve

    Returns:
        List of top n_elite genomes (deep copies)
    """
    # Sort by fitness descending
    sorted_pop = sorted(
        population,
        key=lambda g: weighted_fitness(g.fitness) if g.fitness else 0.0,
        reverse=True
    )

    # Return copies of elites
    return [g.copy() for g in sorted_pop[:n_elite]]


def rank_selection(
    population: List[Genome],
    n_select: int,
) -> List[Genome]:
    """
    Rank-based selection (linear ranking).

    Selection probability is proportional to rank, not raw fitness.
    This reduces selection pressure compared to fitness-proportionate.

    Args:
        population: Current population with fitness scores
        n_select: Number of individuals to select

    Returns:
        List of selected genomes
    """
    # Sort by fitness ascending (worst first)
    sorted_pop = sorted(
        population,
        key=lambda g: weighted_fitness(g.fitness) if g.fitness else 0.0
    )

    # Assign ranks (1 = worst, n = best)
    n = len(sorted_pop)
    ranks = list(range(1, n + 1))

    # Selection probability proportional to rank
    total_rank = sum(ranks)
    probabilities = [r / total_rank for r in ranks]

    # Select with replacement
    selected_indices = np.random.choice(
        n, size=n_select, replace=True, p=probabilities
    )

    return [sorted_pop[i] for i in selected_indices]


# =============================================================================
# Crossover Operators
# =============================================================================

def layer_crossover(
    parent1: Genome,
    parent2: Genome,
    generation: int,
) -> Tuple[Genome, Genome]:
    """
    Single-point layer crossover.

    Swaps layer sequences at a random crossover point.

    Example:
        Parent 1: [16, 8, 4] with ['relu', 'relu', 'relu']
        Parent 2: [8, 16, 8, 8] with ['tanh', 'tanh', 'tanh', 'tanh']
        Crossover at 2:
        Child 1: [16, 8, 8, 8] with ['relu', 'relu', 'tanh', 'tanh']
        Child 2: [8, 16, 4] with ['tanh', 'tanh', 'relu']

    Args:
        parent1: First parent genome
        parent2: Second parent genome
        generation: Generation number for children

    Returns:
        Tuple of two child genomes
    """
    # Determine crossover point
    min_depth = min(len(parent1.hidden_layers), len(parent2.hidden_layers))
    if min_depth < 2:
        # Can't do meaningful crossover, return mutated copies
        return parent1.copy(), parent2.copy()

    crossover_point = random.randint(1, min_depth - 1)

    # Create child layer lists
    child1_layers = (
        parent1.hidden_layers[:crossover_point] +
        parent2.hidden_layers[crossover_point:]
    )
    child2_layers = (
        parent2.hidden_layers[:crossover_point] +
        parent1.hidden_layers[crossover_point:]
    )

    # Create child activation lists
    child1_activations = (
        parent1.activations[:crossover_point] +
        parent2.activations[crossover_point:]
    )
    child2_activations = (
        parent2.activations[:crossover_point] +
        parent1.activations[crossover_point:]
    )

    # Ensure lists match lengths
    child1_activations = _pad_or_trim(child1_activations, len(child1_layers))
    child2_activations = _pad_or_trim(child2_activations, len(child2_layers))

    # Average learning rates
    avg_lr = (parent1.learning_rate + parent2.learning_rate) / 2

    child1 = Genome(
        hidden_layers=child1_layers,
        activations=child1_activations,
        skip_connections=parent1.skip_connections,
        learning_rate=avg_lr,
        genome_id=generate_genome_id(generation, 'cross'),
        generation=generation,
        parents=(parent1.genome_id, parent2.genome_id),
    )

    child2 = Genome(
        hidden_layers=child2_layers,
        activations=child2_activations,
        skip_connections=parent2.skip_connections,
        learning_rate=avg_lr,
        genome_id=generate_genome_id(generation, 'cross'),
        generation=generation,
        parents=(parent1.genome_id, parent2.genome_id),
    )

    return child1, child2


def uniform_crossover(
    parent1: Genome,
    parent2: Genome,
    generation: int,
    swap_prob: float = 0.5,
) -> Tuple[Genome, Genome]:
    """
    Uniform crossover with gene-by-gene swapping.

    Each gene (layer width, activation) is swapped with probability swap_prob.

    Args:
        parent1: First parent genome
        parent2: Second parent genome
        generation: Generation number for children
        swap_prob: Probability of swapping each gene

    Returns:
        Tuple of two child genomes
    """
    min_depth = min(len(parent1.hidden_layers), len(parent2.hidden_layers))

    child1_layers = []
    child2_layers = []
    child1_activations = []
    child2_activations = []

    # Swap genes for shared layers
    for i in range(min_depth):
        if random.random() < swap_prob:
            child1_layers.append(parent2.hidden_layers[i])
            child2_layers.append(parent1.hidden_layers[i])
            child1_activations.append(parent2.activations[i])
            child2_activations.append(parent1.activations[i])
        else:
            child1_layers.append(parent1.hidden_layers[i])
            child2_layers.append(parent2.hidden_layers[i])
            child1_activations.append(parent1.activations[i])
            child2_activations.append(parent2.activations[i])

    # Append remaining layers from longer parent
    if len(parent1.hidden_layers) > min_depth:
        if random.random() < 0.5:
            child1_layers.extend(parent1.hidden_layers[min_depth:])
            child1_activations.extend(parent1.activations[min_depth:])
        else:
            child2_layers.extend(parent1.hidden_layers[min_depth:])
            child2_activations.extend(parent1.activations[min_depth:])

    if len(parent2.hidden_layers) > min_depth:
        if random.random() < 0.5:
            child1_layers.extend(parent2.hidden_layers[min_depth:])
            child1_activations.extend(parent2.activations[min_depth:])
        else:
            child2_layers.extend(parent2.hidden_layers[min_depth:])
            child2_activations.extend(parent2.activations[min_depth:])

    # Handle empty children (shouldn't happen but be safe)
    if not child1_layers:
        child1_layers = parent1.hidden_layers.copy()
        child1_activations = parent1.activations.copy()
    if not child2_layers:
        child2_layers = parent2.hidden_layers.copy()
        child2_activations = parent2.activations.copy()

    # Swap skip connections with probability
    skip1 = parent2.skip_connections if random.random() < swap_prob else parent1.skip_connections
    skip2 = parent1.skip_connections if random.random() < swap_prob else parent2.skip_connections

    # Blend learning rates with some randomness
    lr1 = _blend_lr(parent1.learning_rate, parent2.learning_rate)
    lr2 = _blend_lr(parent2.learning_rate, parent1.learning_rate)

    child1 = Genome(
        hidden_layers=child1_layers,
        activations=child1_activations,
        skip_connections=skip1,
        learning_rate=lr1,
        genome_id=generate_genome_id(generation, 'ucross'),
        generation=generation,
        parents=(parent1.genome_id, parent2.genome_id),
    )

    child2 = Genome(
        hidden_layers=child2_layers,
        activations=child2_activations,
        skip_connections=skip2,
        learning_rate=lr2,
        genome_id=generate_genome_id(generation, 'ucross'),
        generation=generation,
        parents=(parent1.genome_id, parent2.genome_id),
    )

    return child1, child2


# =============================================================================
# Mutation Operators
# =============================================================================

def mutate_genome(
    genome: Genome,
    generation: int,
    mutation_rate: float = 0.2,
    min_depth: int = 1,
    max_depth: int = 10,
    min_width: int = 2,
    max_width: int = 64,
) -> Genome:
    """
    Apply mutations to a genome.

    Mutation types (applied independently):
    - Add layer (10% of mutation_rate)
    - Remove layer (10% of mutation_rate)
    - Resize layer (30% of mutation_rate)
    - Swap activation (30% of mutation_rate)
    - Toggle skip connections (10% of mutation_rate)
    - Perturb learning rate (10% of mutation_rate)

    Args:
        genome: Genome to mutate
        generation: Generation number for the mutated genome
        mutation_rate: Base probability of mutation
        min_depth: Minimum allowed depth
        max_depth: Maximum allowed depth
        min_width: Minimum allowed width
        max_width: Maximum allowed width

    Returns:
        New mutated genome (original is not modified)
    """
    new_layers = genome.hidden_layers.copy()
    new_activations = genome.activations.copy()
    new_skip = genome.skip_connections
    new_lr = genome.learning_rate

    valid_widths = [w for w in WIDTH_CHOICES if min_width <= w <= max_width]

    # Add layer mutation
    if random.random() < mutation_rate * 0.10:
        if len(new_layers) < max_depth:
            insert_pos = random.randint(0, len(new_layers))
            new_width = random.choice(valid_widths)
            new_activation = random.choice(AVAILABLE_ACTIVATIONS)
            new_layers.insert(insert_pos, new_width)
            new_activations.insert(insert_pos, new_activation)

    # Remove layer mutation
    if random.random() < mutation_rate * 0.10:
        if len(new_layers) > min_depth:
            remove_pos = random.randint(0, len(new_layers) - 1)
            new_layers.pop(remove_pos)
            new_activations.pop(remove_pos)

    # Resize layer mutation
    if random.random() < mutation_rate * 0.30:
        if new_layers:
            resize_pos = random.randint(0, len(new_layers) - 1)
            current_width = new_layers[resize_pos]
            # Prefer nearby widths for smoother search
            nearby_widths = [
                w for w in valid_widths
                if 0.5 <= w / current_width <= 2.0
            ]
            if nearby_widths:
                new_layers[resize_pos] = random.choice(nearby_widths)
            else:
                new_layers[resize_pos] = random.choice(valid_widths)

    # Swap activation mutation
    if random.random() < mutation_rate * 0.30:
        if new_activations:
            swap_pos = random.randint(0, len(new_activations) - 1)
            # Exclude current activation for diversity
            other_activations = [
                a for a in AVAILABLE_ACTIVATIONS
                if a != new_activations[swap_pos]
            ]
            if other_activations:
                new_activations[swap_pos] = random.choice(other_activations)

    # Toggle skip connections mutation
    if random.random() < mutation_rate * 0.10:
        new_skip = not new_skip

    # Perturb learning rate mutation (log-scale)
    if random.random() < mutation_rate * 0.10:
        log_lr = np.log10(new_lr)
        log_lr += random.gauss(0, 0.3)  # ~0.3 log-units std dev
        new_lr = float(np.clip(10 ** log_lr, 0.001, 1.0))

    return Genome(
        hidden_layers=new_layers,
        activations=new_activations,
        skip_connections=new_skip,
        learning_rate=new_lr,
        genome_id=generate_genome_id(generation, 'mut'),
        generation=generation,
        parents=(genome.genome_id, 'mutation'),
    )


def gaussian_mutation(
    genome: Genome,
    generation: int,
    sigma: float = 0.2,
) -> Genome:
    """
    Gaussian mutation for continuous parameters.

    Only mutates continuous values (widths, learning rate) using
    Gaussian perturbation. Useful for fine-tuning.

    Args:
        genome: Genome to mutate
        generation: Generation number
        sigma: Standard deviation for Gaussian noise

    Returns:
        New mutated genome
    """
    new_layers = []
    for width in genome.hidden_layers:
        # Perturb width by factor, keeping in valid range
        factor = 1.0 + random.gauss(0, sigma)
        new_width = int(round(width * factor))
        # Snap to nearest valid width
        new_width = min(WIDTH_CHOICES, key=lambda w: abs(w - new_width))
        new_layers.append(new_width)

    # Perturb learning rate
    log_lr = np.log10(genome.learning_rate)
    log_lr += random.gauss(0, sigma)
    new_lr = float(np.clip(10 ** log_lr, 0.001, 1.0))

    return Genome(
        hidden_layers=new_layers,
        activations=genome.activations.copy(),
        skip_connections=genome.skip_connections,
        learning_rate=new_lr,
        genome_id=generate_genome_id(generation, 'gmut'),
        generation=generation,
        parents=(genome.genome_id, 'gaussian_mutation'),
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _pad_or_trim(activations: List[str], target_length: int) -> List[str]:
    """Ensure activation list matches target length."""
    if len(activations) == target_length:
        return activations
    if len(activations) > target_length:
        return activations[:target_length]
    # Pad by repeating last activation
    last = activations[-1] if activations else 'relu'
    return activations + [last] * (target_length - len(activations))


def _blend_lr(lr1: float, lr2: float) -> float:
    """Blend two learning rates with some randomness."""
    # Geometric mean with noise
    log_lr1 = np.log10(lr1)
    log_lr2 = np.log10(lr2)
    blended = (log_lr1 + log_lr2) / 2 + random.gauss(0, 0.1)
    return float(np.clip(10 ** blended, 0.001, 1.0))
