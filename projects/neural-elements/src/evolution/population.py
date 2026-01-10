"""
Population management for evolutionary search.

Handles:
- Initial population creation (seeded + random)
- Diversity preservation
- Loading top performers from previous phases
"""

import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from .genome import (
    Genome,
    create_random_genome,
    genome_from_element_config,
    AVAILABLE_ACTIVATIONS,
)
from .fitness import weighted_fitness


def create_initial_population(
    population_size: int = 50,
    phase_winners: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    seed_ratio: float = 0.30,
    min_depth: int = 1,
    max_depth: int = 6,
    min_width: int = 2,
    max_width: int = 64,
) -> List[Genome]:
    """
    Create initial population with mix of seeded and random individuals.

    Composition (default):
    - 30% seeded from Phase 3-8 top performers
    - 70% random (with architecture constraints)

    Args:
        population_size: Total population size
        phase_winners: Top configs from previous phases
        seed: Random seed for reproducibility
        seed_ratio: Fraction of population to seed from winners
        min_depth: Minimum depth for random genomes
        max_depth: Maximum depth for random genomes
        min_width: Minimum width for random genomes
        max_width: Maximum width for random genomes

    Returns:
        List of Genome objects forming the initial population
    """
    if seed is not None:
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)

    population = []

    # Seed from phase winners
    if phase_winners:
        n_seeded = min(int(population_size * seed_ratio), len(phase_winners))
        for i, config in enumerate(phase_winners[:n_seeded]):
            genome = genome_from_element_config(
                config,
                generation=0,
                prefix=f'seed{i:02d}',
            )
            population.append(genome)

    # Fill rest with random individuals
    while len(population) < population_size:
        genome = create_random_genome(
            generation=0,
            min_depth=min_depth,
            max_depth=max_depth,
            min_width=min_width,
            max_width=max_width,
            prefix='rand',
        )
        population.append(genome)

    return population


def get_phase_winners(data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load top-performing configurations from Phases 3-8.

    Attempts to load from aggregated phase summaries. Falls back to
    hardcoded defaults based on README findings.

    Args:
        data_dir: Path to data directory (optional)

    Returns:
        List of element configurations sorted by performance
    """
    winners = []

    # Try to load from phase summary files
    if data_dir:
        data_path = Path(data_dir)
        for phase in [3, 4, 5, 6, 7, 8]:
            summary_file = data_path / f'phase{phase}_summary.csv'
            if summary_file.exists():
                loaded = _load_winners_from_csv(summary_file)
                winners.extend(loaded)

    # If no files found, use hardcoded winners from README
    if not winners:
        winners = _get_hardcoded_winners()

    # Deduplicate by architecture signature
    seen = set()
    unique_winners = []
    for w in winners:
        sig = _config_signature(w)
        if sig not in seen:
            seen.add(sig)
            unique_winners.append(w)

    return unique_winners


def _get_hardcoded_winners() -> List[Dict[str, Any]]:
    """
    Hardcoded top performers from Phase 3-8 findings.

    Based on README experimental results.
    """
    return [
        # Phase 3: Top activations at various depths
        {'hidden_layers': [8, 8, 8], 'activation': 'leaky_relu'},
        {'hidden_layers': [8, 8, 8], 'activation': 'relu'},
        {'hidden_layers': [8, 8, 8], 'activation': 'sine'},
        {'hidden_layers': [8, 8, 8], 'activation': 'tanh'},
        {'hidden_layers': [8, 8, 8], 'activation': 'gelu'},
        {'hidden_layers': [8], 'activation': 'gelu'},  # Best for XOR
        {'hidden_layers': [8, 8, 8, 8, 8], 'activation': 'tanh'},  # Best for moons
        {'hidden_layers': [8, 8, 8, 8], 'activation': 'leaky_relu'},  # Best for circles

        # Phase 4: Sine at extreme depths (exceptional stability)
        {'hidden_layers': [8] * 6, 'activation': 'sine'},
        {'hidden_layers': [8] * 8, 'activation': 'sine'},
        {'hidden_layers': [8] * 10, 'activation': 'sine'},

        # Phase 4: Skip connections help sigmoid
        {'hidden_layers': [8] * 6, 'activation': 'sigmoid', 'skip_connections': True},

        # Phase 5: Bottleneck architectures
        {'hidden_layers': [32, 8, 32], 'activation': 'relu'},
        {'hidden_layers': [16, 4, 16], 'activation': 'leaky_relu'},

        # Phase 5: Medium balanced (best parameter efficiency)
        {'hidden_layers': [12, 12], 'activation': 'relu'},
        {'hidden_layers': [12, 12], 'activation': 'tanh'},

        # Phase 5: Pyramid patterns
        {'hidden_layers': [4, 8, 16], 'activation': 'relu'},
        {'hidden_layers': [16, 8, 4], 'activation': 'leaky_relu'},

        # Phase 6: Fast convergence
        {'hidden_layers': [8, 8], 'activation': 'leaky_relu'},

        # Phase 7: Best generalizers
        {'hidden_layers': [8], 'activation': 'sine'},
        {'hidden_layers': [8], 'activation': 'relu'},
        {'hidden_layers': [8, 8], 'activation': 'tanh'},

        # Diverse depth variations
        {'hidden_layers': [16], 'activation': 'relu'},
        {'hidden_layers': [8, 8, 8, 8], 'activation': 'relu'},
        {'hidden_layers': [4, 8, 4], 'activation': 'sine'},
    ]


def _load_winners_from_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Load top configs from a phase summary CSV."""
    from .genome import AVAILABLE_ACTIVATIONS

    # Map for common activation name issues in CSV data
    ACTIVATION_FIXES = {
        'gel': 'gelu',
        'leaky': 'leaky_relu',
        'leakyrelu': 'leaky_relu',
    }

    winners = []
    try:
        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Sort by accuracy descending (handle both 'accuracy' and 'mean_accuracy')
        def get_accuracy(r):
            return float(r.get('mean_accuracy', r.get('accuracy', 0)))
        rows.sort(key=get_accuracy, reverse=True)

        # Take top 5 from each phase
        for row in rows[:5]:
            activation = row.get('activation', 'relu')

            # Fix common activation name issues
            activation = ACTIVATION_FIXES.get(activation, activation)

            # Skip if activation is not recognized
            if activation not in AVAILABLE_ACTIVATIONS:
                continue

            # Parse hidden layers from depth/width or architecture column
            if 'architecture' in row:
                hidden_layers = _parse_layers(row.get('architecture', '[8]'))
            elif 'depth' in row and 'width' in row:
                depth = int(row.get('depth', 1))
                width = int(row.get('width', 8))
                hidden_layers = [width] * depth
            else:
                hidden_layers = [8]

            config = {
                'hidden_layers': hidden_layers,
                'activation': activation,
            }
            if row.get('skip_connections', 'False').lower() == 'true':
                config['skip_connections'] = True
            winners.append(config)

    except Exception:
        pass  # Fall back to hardcoded if CSV parsing fails

    return winners


def _parse_layers(layer_str: str) -> List[int]:
    """Parse layer string like '[8, 8, 8]' to list."""
    try:
        # Handle various formats
        layer_str = layer_str.strip('[]').replace(' ', '')
        if not layer_str:
            return [8]
        return [int(x) for x in layer_str.split(',')]
    except Exception:
        return [8]  # Default fallback


def _config_signature(config: Dict[str, Any]) -> str:
    """Create unique signature for deduplication."""
    layers = tuple(config.get('hidden_layers', [8]))
    act = config.get('activation', 'relu')
    skip = config.get('skip_connections', False)
    return f"{layers}_{act}_{skip}"


def compute_genome_distance(g1: Genome, g2: Genome) -> float:
    """
    Compute structural distance between two genomes.

    Used for diversity preservation. Higher distance = more different.

    Components:
    - Depth difference (normalized)
    - Width similarity for shared layers
    - Activation similarity
    - Skip connection difference

    Args:
        g1: First genome
        g2: Second genome

    Returns:
        Distance in [0, 1] range
    """
    # Depth difference (normalized by max reasonable depth)
    depth_diff = abs(len(g1.hidden_layers) - len(g2.hidden_layers)) / 10.0

    # Width and activation similarity for shared layers
    min_depth = min(len(g1.hidden_layers), len(g2.hidden_layers))
    if min_depth > 0:
        # Width difference (normalized by max width)
        width_diff = sum(
            abs(g1.hidden_layers[i] - g2.hidden_layers[i]) / 64.0
            for i in range(min_depth)
        ) / min_depth

        # Activation difference (0 or 1 per layer)
        activation_diff = sum(
            0 if g1.activations[i] == g2.activations[i] else 1
            for i in range(min_depth)
        ) / min_depth
    else:
        width_diff = 1.0
        activation_diff = 1.0

    # Skip connection difference
    skip_diff = 0.0 if g1.skip_connections == g2.skip_connections else 1.0

    # Learning rate difference (log scale)
    import numpy as np
    lr_diff = abs(np.log10(g1.learning_rate) - np.log10(g2.learning_rate)) / 3.0

    # Weighted combination
    distance = (
        0.25 * depth_diff +
        0.25 * width_diff +
        0.25 * activation_diff +
        0.15 * skip_diff +
        0.10 * min(lr_diff, 1.0)
    )

    return min(distance, 1.0)


def enforce_diversity(
    population: List[Genome],
    min_distance: float = 0.15,
    max_removals: Optional[int] = None,
) -> List[Genome]:
    """
    Remove near-duplicates to maintain population diversity.

    Keeps the fitter individual when two are too similar.

    Args:
        population: List of genomes (should have fitness evaluated)
        min_distance: Minimum distance threshold
        max_removals: Maximum genomes to remove (default: 10% of population)

    Returns:
        Filtered population with near-duplicates removed
    """
    if max_removals is None:
        max_removals = max(1, len(population) // 10)

    # Sort by fitness descending (keep best when removing duplicates)
    sorted_pop = sorted(
        population,
        key=lambda g: weighted_fitness(g.fitness) if g.fitness else 0.0,
        reverse=True
    )

    diverse_population = []
    removals = 0

    for genome in sorted_pop:
        is_unique = True

        # Check distance to all already-kept genomes
        for existing in diverse_population:
            if compute_genome_distance(genome, existing) < min_distance:
                is_unique = False
                break

        if is_unique:
            diverse_population.append(genome)
        else:
            removals += 1
            if removals >= max_removals:
                # Keep remaining even if similar
                diverse_population.extend(sorted_pop[len(diverse_population) + removals:])
                break

    return diverse_population


def get_population_stats(population: List[Genome]) -> Dict[str, Any]:
    """
    Compute statistics about the population.

    Args:
        population: List of genomes

    Returns:
        Dictionary with population statistics
    """
    if not population:
        return {'size': 0}

    depths = [g.depth for g in population]
    params = [g.total_params for g in population]
    lrs = [g.learning_rate for g in population]

    # Count activations
    activation_counts = {}
    for g in population:
        for act in g.activations:
            activation_counts[act] = activation_counts.get(act, 0) + 1

    # Fitness stats (if evaluated)
    evaluated = [g for g in population if g.fitness is not None]
    if evaluated:
        fitnesses = [weighted_fitness(g.fitness) for g in evaluated]
        fitness_stats = {
            'min_fitness': min(fitnesses),
            'max_fitness': max(fitnesses),
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'evaluated_count': len(evaluated),
        }
    else:
        fitness_stats = {'evaluated_count': 0}

    import numpy as np
    return {
        'size': len(population),
        'depth_range': (min(depths), max(depths)),
        'mean_depth': np.mean(depths),
        'params_range': (min(params), max(params)),
        'mean_params': np.mean(params),
        'lr_range': (min(lrs), max(lrs)),
        'activation_counts': activation_counts,
        'skip_count': sum(1 for g in population if g.skip_connections),
        **fitness_stats,
    }
