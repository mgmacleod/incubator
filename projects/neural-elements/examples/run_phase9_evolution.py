#!/usr/bin/env python3
"""
Phase 9: Evolutionary Discovery Training Run

Evolves neural element architectures using genetic algorithms to discover
optimal configurations beyond the grid-based exploration of Phases 3-8.

Usage:
    python examples/run_phase9_evolution.py [options]

Options:
    --population N      Population size (default: 50)
    --generations N     Number of generations (default: 30)
    --workers N         Parallel workers (default: cpu_count - 1)
    --seed N            Random seed for reproducibility
    --resume PATH       Resume from checkpoint file
    --datasets D [D...] Datasets to optimize for (default: all 4)
    --quick             Use reduced epochs for quick testing
    --no-confirm        Skip confirmation prompt
"""

import argparse
import sys
import time
from pathlib import Path
from multiprocessing import cpu_count

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.config_selection import Phase9Config, estimate_runtime
from src.evolution.engine import EvolutionEngine, EvolutionConfig
from src.evolution.population import get_phase_winners
from src.datasets.toy import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Phase 9 evolutionary discovery'
    )
    parser.add_argument(
        '--population', type=int, default=50,
        help='Population size (default: 50)'
    )
    parser.add_argument(
        '--generations', type=int, default=30,
        help='Number of generations (default: 30)'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of parallel workers (default: cpu_count - 1)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint file to resume from'
    )
    parser.add_argument(
        '--datasets', nargs='+',
        default=['xor', 'moons', 'circles', 'spirals'],
        help='Datasets to optimize for'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Use reduced epochs for quick testing'
    )
    parser.add_argument(
        '--no-confirm', action='store_true',
        help='Skip confirmation prompt'
    )
    return parser.parse_args()


def print_banner():
    print("=" * 70)
    print("   NEURAL ELEMENTS - Phase 9: Evolutionary Discovery")
    print("=" * 70)


def print_config(config: Phase9Config, n_workers: int, datasets: list):
    print("\nConfiguration:")
    print(f"   Population size:    {config.population_size}")
    print(f"   Generations:        {config.n_generations}")
    print(f"   Elite count:        {config.n_elite}")
    print(f"   Tournament size:    {config.tournament_size}")
    print(f"   Crossover rate:     {config.crossover_rate}")
    print(f"   Mutation rate:      {config.mutation_rate}")
    print(f"   Workers:            {n_workers}")
    print(f"   Datasets:           {', '.join(datasets)}")

    print("\nTraining parameters:")
    print(f"   Quick epochs:       {config.quick_epochs}")
    print(f"   Full epochs:        {config.full_epochs}")
    print(f"   Quick threshold:    {config.quick_accuracy_threshold}")

    print("\nArchitecture constraints:")
    print(f"   Depth range:        [{config.min_depth}, {config.max_depth}]")
    print(f"   Width range:        [{config.min_width}, {config.max_width}]")

    # Estimate runtime
    estimated_evals = config.get_estimated_experiments()
    runtime = estimate_runtime(estimated_evals, n_workers, 0.4)  # ~0.4s per eval

    print(f"\nEstimated scale:")
    print(f"   ~{estimated_evals:,} fitness evaluations")
    print(f"   ~{runtime['total_minutes']:.1f} minutes runtime")


def progress_callback(gen: int, total: int, stats: dict):
    """Print progress during evolution."""
    pct = 100 * gen / total
    fitness = stats.get('best_fitness', 0)
    evals = stats.get('evaluations', 0)
    print(
        f"\r   Gen {gen:3d}/{total} ({pct:5.1f}%) | "
        f"Best fitness: {fitness:.4f} | "
        f"Evaluations: {evals:,}",
        end='', flush=True
    )


def run_evolution_for_dataset(
    dataset_name: str,
    config: Phase9Config,
    n_workers: int,
    seed: int = None,
    resume_path: str = None,
):
    """Run evolution for a single dataset."""
    print(f"\n{'='*60}")
    print(f"   Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Get dataset
    X, y = get_dataset(dataset_name, n_samples=200)

    # Split into train/test (80/20)
    n_train = int(0.8 * len(X))
    indices = list(range(len(X)))
    if seed is not None:
        import random
        random.seed(seed)
        random.shuffle(indices)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Create evolution config
    evo_config = EvolutionConfig(
        population_size=config.population_size,
        n_elite=config.n_elite,
        tournament_size=config.tournament_size,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        quick_epochs=config.quick_epochs,
        full_epochs=config.full_epochs,
        quick_accuracy_threshold=config.quick_accuracy_threshold,
        min_depth=config.min_depth,
        max_depth=config.max_depth,
        min_width=config.min_width,
        max_width=config.max_width,
        min_genome_distance=config.min_genome_distance,
        early_stop_patience=config.early_stop_patience,
        early_stop_min_improvement=config.early_stop_min_improvement,
        checkpoint_every=config.checkpoint_every,
        checkpoint_dir=str(project_root / 'data' / 'evolution'),
        n_workers=n_workers,
    )

    # Create engine
    engine = EvolutionEngine(
        config=evo_config,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name=dataset_name,
    )

    # Initialize or resume
    if resume_path:
        print(f"   Resuming from: {resume_path}")
        engine.load_checkpoint(Path(resume_path))
        print(f"   Resumed at generation {engine.generation}")
    else:
        print("   Initializing population...")
        phase_winners = get_phase_winners(str(project_root / 'data'))
        engine.initialize_population(phase_winners=phase_winners, seed=seed)
        print(f"   Population initialized: {len(engine.population)} genomes")
        print(f"   Seeded from {len(phase_winners)} phase winners")

    # Run evolution
    print(f"\n   Starting evolution...")
    start_time = time.time()

    result = engine.evolve(
        n_generations=config.n_generations,
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time
    print()  # New line after progress

    # Print results
    print(f"\n   Results:")
    print(f"   --------")
    print(f"   Generations completed: {result.generations_completed}")
    print(f"   Total evaluations:     {result.total_evaluations:,}")
    print(f"   Best fitness:          {result.best_fitness:.4f}")
    print(f"   Best accuracy:         {result.best_accuracy:.4f}")
    print(f"   Runtime:               {elapsed:.1f}s")

    if result.early_stopped:
        print(f"   Early stopped:         Yes ({result.early_stop_reason})")

    # Print champions
    print(f"\n   Top 5 Champions:")
    for i, champion in enumerate(result.champions[:5], 1):
        fitness = champion.fitness
        print(
            f"   {i}. {champion.architecture_string:25s} | "
            f"Acc: {fitness.test_accuracy:.3f} | "
            f"Params: {champion.total_params:4d}"
        )

    return result


def main():
    args = parse_args()

    print_banner()

    # Create config
    config = Phase9Config(
        population_size=args.population,
        n_generations=args.generations,
        datasets=args.datasets,
    )

    # Apply quick mode
    if args.quick:
        config.quick_epochs = 25
        config.full_epochs = 100
        config.n_generations = min(config.n_generations, 10)

    # Determine workers
    n_workers = args.workers or max(1, cpu_count() - 1)

    # Print config
    print_config(config, n_workers, args.datasets)

    # Confirmation
    if not args.no_confirm:
        print()
        response = input("Proceed with evolution? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    # Run evolution for each dataset
    all_results = {}
    start_time = time.time()

    for dataset in args.datasets:
        result = run_evolution_for_dataset(
            dataset_name=dataset,
            config=config,
            n_workers=n_workers,
            seed=args.seed,
            resume_path=args.resume,
        )
        all_results[dataset] = result

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("   EVOLUTION COMPLETE - Summary")
    print("=" * 70)
    print(f"\n   Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\n   Best architectures by dataset:")

    for dataset, result in all_results.items():
        best = result.get_best_genome()
        if best:
            print(
                f"   {dataset:8s}: {best.architecture_string:25s} | "
                f"Acc: {best.fitness.test_accuracy:.3f}"
            )

    print("\n   Checkpoints saved to: data/evolution/")
    print("   Run visualize_phase9.py to generate charts.")


if __name__ == '__main__':
    main()
