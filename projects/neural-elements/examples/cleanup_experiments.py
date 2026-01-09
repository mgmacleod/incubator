#!/usr/bin/env python3
"""
Utility script to clean up experiment data.

Use this when:
1. You want to start fresh with Phase 4
2. There are stale pending experiments from failed runs
3. The index is out of sync with actual experiment files

Options:
- --rebuild-index: Rebuild index.json from experiment directories
- --clear-pending: Remove all pending experiments
- --clear-phase4: Remove all Phase 4 experiments (depths 6,7,8,10)
- --clear-all: Remove ALL experiments (use with caution!)
- --status: Just show status without changing anything
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.persistence import ExperimentStore
from src.analysis.aggregator import _parse_element_name


def get_status(store: ExperimentStore) -> dict:
    """Get current experiment status."""
    stats = store.get_statistics()

    all_exps = store.list_experiments(limit=100000)
    phase4_depths = {6, 7, 8, 10}

    # Categorize experiments
    phase3_exps = []
    phase4_exps = []
    other_exps = []

    for exp in all_exps:
        parsed = _parse_element_name(exp.get('element_name', ''))
        if parsed:
            if parsed['depth'] in phase4_depths:
                phase4_exps.append(exp)
            elif parsed['depth'] in {1, 2, 3, 4, 5}:
                phase3_exps.append(exp)
            else:
                other_exps.append(exp)
        else:
            other_exps.append(exp)

    return {
        'total': stats['total'],
        'by_status': stats['by_status'],
        'phase3': {
            'total': len(phase3_exps),
            'completed': len([e for e in phase3_exps if e.get('status') == 'completed']),
            'pending': len([e for e in phase3_exps if e.get('status') == 'pending']),
        },
        'phase4': {
            'total': len(phase4_exps),
            'completed': len([e for e in phase4_exps if e.get('status') == 'completed']),
            'pending': len([e for e in phase4_exps if e.get('status') == 'pending']),
        },
        'other': len(other_exps),
    }


def rebuild_index(store: ExperimentStore) -> int:
    """Rebuild index.json by scanning experiment directories."""
    experiments_dir = store.experiments_dir

    new_index = {'version': '1.0', 'experiments': {}}
    count = 0
    fixed = 0

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        metadata_file = exp_dir / 'metadata.json'
        result_file = exp_dir / 'result.json'

        if not metadata_file.exists():
            continue

        try:
            metadata = json.loads(metadata_file.read_text())
            exp_id = metadata.get('experiment_id', exp_dir.name)

            # Determine actual status based on files
            if result_file.exists():
                result = json.loads(result_file.read_text())
                actual_status = 'completed'

                # Update metadata if it says pending but result exists
                if metadata.get('status') == 'pending':
                    metadata['status'] = 'completed'
                    metadata_file.write_text(json.dumps(metadata, indent=2))
                    fixed += 1

                new_index['experiments'][exp_id] = {
                    'experiment_id': exp_id,
                    'element_name': metadata.get('element_name', ''),
                    'dataset': metadata.get('dataset_name', ''),
                    'status': 'completed',
                    'final_accuracy': result.get('final_accuracy'),
                    'final_loss': result.get('final_loss'),
                    'training_time': result.get('training_time_seconds'),
                    'created_at': metadata.get('created_at'),
                    'job_id': metadata.get('job_id'),
                }
            else:
                new_index['experiments'][exp_id] = {
                    'experiment_id': exp_id,
                    'element_name': metadata.get('element_name', ''),
                    'dataset': metadata.get('dataset_name', ''),
                    'status': metadata.get('status', 'pending'),
                    'created_at': metadata.get('created_at'),
                    'job_id': metadata.get('job_id'),
                }

            count += 1

        except Exception as e:
            print(f"  Warning: Error processing {exp_dir}: {e}")

    # Write new index
    store.index_file.write_text(json.dumps(new_index, indent=2))

    print(f"  Rebuilt index with {count} experiments")
    if fixed > 0:
        print(f"  Fixed {fixed} experiments that had results but were marked pending")

    return count


def clear_pending(store: ExperimentStore) -> int:
    """Remove all pending experiments."""
    all_exps = store.list_experiments(status='pending', limit=100000)
    count = 0

    for exp in all_exps:
        exp_id = exp.get('experiment_id')
        if exp_id:
            exp_dir = store.experiments_dir / exp_id
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            count += 1

    # Rebuild index after deletion
    rebuild_index(store)

    return count


def clear_phase4(store: ExperimentStore) -> int:
    """Remove all Phase 4 experiments (depths 6,7,8,10)."""
    all_exps = store.list_experiments(limit=100000)
    phase4_depths = {6, 7, 8, 10}
    count = 0

    for exp in all_exps:
        parsed = _parse_element_name(exp.get('element_name', ''))
        if parsed and parsed['depth'] in phase4_depths:
            exp_id = exp.get('experiment_id')
            if exp_id:
                exp_dir = store.experiments_dir / exp_id
                if exp_dir.exists():
                    shutil.rmtree(exp_dir)
                count += 1

    # Rebuild index after deletion
    rebuild_index(store)

    return count


def clear_all(store: ExperimentStore) -> int:
    """Remove ALL experiments."""
    count = 0

    for exp_dir in store.experiments_dir.iterdir():
        if exp_dir.is_dir():
            shutil.rmtree(exp_dir)
            count += 1

    # Reset index
    store.index_file.write_text(json.dumps({'version': '1.0', 'experiments': {}}, indent=2))

    # Clear jobs
    jobs_file = store.base_path / 'jobs.json'
    if jobs_file.exists():
        jobs_file.write_text(json.dumps({}, indent=2))

    return count


def main():
    parser = argparse.ArgumentParser(description='Clean up experiment data')
    parser.add_argument('--rebuild-index', action='store_true',
                        help='Rebuild index.json from experiment directories')
    parser.add_argument('--clear-pending', action='store_true',
                        help='Remove all pending experiments')
    parser.add_argument('--clear-phase4', action='store_true',
                        help='Remove all Phase 4 experiments')
    parser.add_argument('--clear-all', action='store_true',
                        help='Remove ALL experiments (use with caution!)')
    parser.add_argument('--status', action='store_true',
                        help='Just show status')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    store = ExperimentStore(str(data_path))

    # Always show status first
    print("=" * 60)
    print("EXPERIMENT STORE STATUS")
    print("=" * 60)

    status = get_status(store)
    print(f"\nTotal experiments: {status['total']}")
    print(f"\nBy status:")
    for s, count in sorted(status['by_status'].items()):
        print(f"  {s}: {count}")

    print(f"\nPhase 3 (depths 1-5):")
    print(f"  Total: {status['phase3']['total']}")
    print(f"  Completed: {status['phase3']['completed']}")
    print(f"  Pending: {status['phase3']['pending']}")

    print(f"\nPhase 4 (depths 6,7,8,10):")
    print(f"  Total: {status['phase4']['total']}")
    print(f"  Completed: {status['phase4']['completed']}")
    print(f"  Pending: {status['phase4']['pending']}")

    if status['other'] > 0:
        print(f"\nOther: {status['other']}")

    if args.status:
        return

    # Perform actions
    print("\n" + "=" * 60)

    if args.clear_all:
        response = input("WARNING: This will delete ALL experiments. Are you sure? [yes/N]: ")
        if response.strip().lower() == 'yes':
            count = clear_all(store)
            print(f"Removed {count} experiments")
        else:
            print("Aborted")

    elif args.clear_phase4:
        response = input(f"This will delete {status['phase4']['total']} Phase 4 experiments. Continue? [y/N]: ")
        if response.strip().lower() == 'y':
            count = clear_phase4(store)
            print(f"Removed {count} Phase 4 experiments")
        else:
            print("Aborted")

    elif args.clear_pending:
        pending_count = status['by_status'].get('pending', 0)
        response = input(f"This will delete {pending_count} pending experiments. Continue? [y/N]: ")
        if response.strip().lower() == 'y':
            count = clear_pending(store)
            print(f"Removed {count} pending experiments")
        else:
            print("Aborted")

    elif args.rebuild_index:
        print("Rebuilding index from experiment directories...")
        rebuild_index(store)

    else:
        print("No action specified. Use --help to see options.")
        print("\nCommon workflows:")
        print("  1. Fix out-of-sync index: python cleanup_experiments.py --rebuild-index")
        print("  2. Start Phase 4 fresh:   python cleanup_experiments.py --clear-phase4")
        print("  3. Remove stale pending:  python cleanup_experiments.py --clear-pending")


if __name__ == '__main__':
    main()
