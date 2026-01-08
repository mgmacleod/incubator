"""
Flask Blueprint for bulk training API endpoints.

Provides endpoints for:
- Submitting bulk training jobs
- Monitoring job progress
- Retrieving experiment results
- Loading saved weights
"""

from flask import Blueprint, jsonify, request, current_app
import numpy as np

from ..core.elements import NeuralElement
from ..core.jobs import JobStatus
from ..visualization.interactive import generate_decision_boundary_data


bulk_bp = Blueprint('bulk', __name__, url_prefix='/api/bulk')


def get_job_manager():
    """Get the JobManager from app config."""
    return current_app.config['JOB_MANAGER']


def get_experiment_store():
    """Get the ExperimentStore from app config."""
    return current_app.config['EXPERIMENT_STORE']


@bulk_bp.route('/jobs', methods=['POST'])
def create_bulk_job():
    """
    Create a new bulk training job.

    Request body:
    {
        "elements": [
            {"hidden_layers": [4], "activation": "relu"},
            {"hidden_layers": [4, 4], "activation": "relu"},
            ...
        ],
        "datasets": ["spirals", "xor", "moons"],
        "training": {
            "epochs": 500,
            "learning_rate": 0.1,
            "batch_size": null,
            "record_every": 50
        },
        "n_trials": 1
    }

    Response:
    {
        "job_id": "job_20240108_143052_abc123",
        "status": "running",
        "total_runs": 27,
        "message": "Job submitted successfully"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body required'}), 400

    element_configs = data.get('elements', [])
    dataset_names = data.get('datasets', [])
    training_config = data.get('training', {})
    n_trials = data.get('n_trials', 1)

    if not element_configs:
        return jsonify({'error': 'At least one element configuration required'}), 400

    if not dataset_names:
        return jsonify({'error': 'At least one dataset required'}), 400

    # Set default training config values
    training_config.setdefault('epochs', 500)
    training_config.setdefault('learning_rate', 0.1)
    training_config.setdefault('record_every', 50)

    try:
        job_manager = get_job_manager()
        job_id = job_manager.submit_bulk_job(
            element_configs=element_configs,
            dataset_names=dataset_names,
            training_config=training_config,
            n_trials=n_trials,
        )

        job = job_manager.get_job_status(job_id)

        return jsonify({
            'job_id': job_id,
            'status': job.status.value,
            'total_runs': job.total_runs,
            'message': 'Job submitted successfully',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bulk_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """
    List all jobs with optional status filter.

    Query params:
        status: Filter by status (pending, running, completed, failed, cancelled)
        limit: Max number of jobs to return (default 50)

    Response:
    {
        "jobs": [...],
        "total": 15
    }
    """
    status_filter = request.args.get('status')
    limit = int(request.args.get('limit', 50))

    job_manager = get_job_manager()

    status = None
    if status_filter:
        try:
            status = JobStatus(status_filter)
        except ValueError:
            return jsonify({'error': f'Invalid status: {status_filter}'}), 400

    jobs = job_manager.list_jobs(status=status, limit=limit)

    return jsonify({
        'jobs': [job.to_dict() for job in jobs],
        'total': len(jobs),
    })


@bulk_bp.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id: str):
    """
    Get detailed status of a specific job.

    Response:
    {
        "job_id": "...",
        "status": "running",
        "progress": {
            "completed": 12,
            "failed": 1,
            "total": 27
        },
        "experiment_ids": ["exp_...", ...],
        "errors": [{"experiment_id": "...", "error": "..."}],
        ...
    }
    """
    job_manager = get_job_manager()
    job = job_manager.get_job_status(job_id)

    if job is None:
        return jsonify({'error': 'Job not found'}), 404

    response = job.to_dict()
    response['progress'] = {
        'completed': job.completed_runs,
        'failed': job.failed_runs,
        'total': job.total_runs,
        'percent': round(100 * (job.completed_runs + job.failed_runs) / max(1, job.total_runs), 1),
    }

    return jsonify(response)


@bulk_bp.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id: str):
    """Cancel a running job."""
    job_manager = get_job_manager()
    success = job_manager.cancel_job(job_id)

    if success:
        return jsonify({'success': True, 'message': 'Job cancelled'})
    else:
        return jsonify({'success': False, 'message': 'Job not running or not found'}), 400


@bulk_bp.route('/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id: str):
    """
    Get results summary for a completed job.

    Response:
    {
        "job_id": "...",
        "results": [
            {
                "experiment_id": "...",
                "element_name": "REL-2x4",
                "dataset": "spirals",
                "final_accuracy": 0.92,
                "final_loss": 0.15,
                "training_time": 2.34
            },
            ...
        ],
        "summary": {
            "best_by_dataset": {...},
            "best_overall": {...}
        }
    }
    """
    job_manager = get_job_manager()
    job = job_manager.get_job_status(job_id)

    if job is None:
        return jsonify({'error': 'Job not found'}), 404

    results = job_manager.get_job_results(job_id)

    # Build summary
    summary = {
        'best_by_dataset': {},
        'best_by_element': {},
        'best_overall': None,
    }

    if results:
        # Best by dataset
        datasets = set(r['dataset'] for r in results)
        for dataset in datasets:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            best = max(dataset_results, key=lambda r: r['final_accuracy'])
            summary['best_by_dataset'][dataset] = {
                'element_name': best['element_name'],
                'accuracy': best['final_accuracy'],
                'experiment_id': best['experiment_id'],
            }

        # Best by element
        elements = set(r['element_name'] for r in results)
        for element in elements:
            element_results = [r for r in results if r['element_name'] == element]
            avg_accuracy = sum(r['final_accuracy'] for r in element_results) / len(element_results)
            summary['best_by_element'][element] = {
                'avg_accuracy': round(avg_accuracy, 4),
                'n_runs': len(element_results),
            }

        # Best overall
        best = max(results, key=lambda r: r['final_accuracy'])
        summary['best_overall'] = {
            'element_name': best['element_name'],
            'dataset': best['dataset'],
            'accuracy': best['final_accuracy'],
            'experiment_id': best['experiment_id'],
        }

    return jsonify({
        'job_id': job_id,
        'status': job.status.value,
        'results': results,
        'summary': summary,
    })


@bulk_bp.route('/experiments', methods=['GET'])
def list_experiments():
    """
    List experiments with optional filtering.

    Query params:
        status: Filter by status
        dataset: Filter by dataset name
        job_id: Filter by job ID
        limit: Max results (default 100)
        offset: Pagination offset
    """
    store = get_experiment_store()

    experiments = store.list_experiments(
        status=request.args.get('status'),
        dataset=request.args.get('dataset'),
        job_id=request.args.get('job_id'),
        limit=int(request.args.get('limit', 100)),
        offset=int(request.args.get('offset', 0)),
    )

    return jsonify({
        'experiments': experiments,
        'total': len(experiments),
    })


@bulk_bp.route('/experiments/<experiment_id>', methods=['GET'])
def get_experiment(experiment_id: str):
    """
    Get full experiment details including weights.

    Response includes full weights, biases, and training history
    for loading a trained model.
    """
    store = get_experiment_store()
    metadata, result = store.load_experiment(experiment_id)

    if metadata is None:
        return jsonify({'error': 'Experiment not found'}), 404

    response = {
        'experiment_id': experiment_id,
        'metadata': {
            'element_name': metadata.element_name,
            'element_config': metadata.element_config,
            'dataset_name': metadata.dataset_name,
            'training_config': metadata.training_config,
            'status': metadata.status,
            'created_at': metadata.created_at,
            'completed_at': metadata.completed_at,
        },
    }

    if result:
        response['result'] = {
            'weights': result.weights,
            'biases': result.biases,
            'history': result.history,
            'final_loss': result.final_loss,
            'final_accuracy': result.final_accuracy,
            'training_time_seconds': result.training_time_seconds,
        }

    return jsonify(response)


@bulk_bp.route('/experiments/<experiment_id>/boundary', methods=['GET'])
def get_experiment_boundary(experiment_id: str):
    """
    Get decision boundary visualization data for a saved experiment.

    Reconstructs the element from saved weights and generates boundary.

    Query params:
        resolution: Grid resolution (default 50)
    """
    store = get_experiment_store()
    metadata, result = store.load_experiment(experiment_id)

    if metadata is None or result is None:
        return jsonify({'error': 'Experiment not found or incomplete'}), 404

    # Reconstruct element with saved weights
    element = NeuralElement(**metadata.element_config)
    element.load_weights(result.weights, result.biases)
    element.history = result.history

    # Generate boundary data
    resolution = int(request.args.get('resolution', 50))
    boundary_data = generate_decision_boundary_data(
        element,
        resolution=resolution,
    )

    return jsonify({
        'experiment_id': experiment_id,
        'element_name': metadata.element_name,
        'dataset': metadata.dataset_name,
        **boundary_data,
    })


@bulk_bp.route('/experiments/<experiment_id>', methods=['DELETE'])
def delete_experiment(experiment_id: str):
    """Delete an experiment."""
    store = get_experiment_store()
    success = store.delete_experiment(experiment_id)

    if success:
        return jsonify({'success': True, 'message': 'Experiment deleted'})
    else:
        return jsonify({'success': False, 'message': 'Experiment not found'}), 404


@bulk_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """Get aggregate statistics about stored experiments."""
    store = get_experiment_store()
    stats = store.get_statistics()

    return jsonify(stats)


@bulk_bp.route('/experiments/<experiment_id>/training-lab', methods=['GET'])
def get_experiment_for_training_lab(experiment_id: str):
    """
    Get experiment data formatted for the Training Lab visualization.

    This returns data in the same format as /api/train, allowing
    saved experiments to be loaded and displayed in the Training Lab.

    Returns the final frame (trained state) plus training history.
    """
    from ..datasets.toy import get_dataset

    store = get_experiment_store()
    metadata, result = store.load_experiment(experiment_id)

    if metadata is None or result is None:
        return jsonify({'error': 'Experiment not found or incomplete'}), 404

    # Reconstruct element with saved weights
    element = NeuralElement(**metadata.element_config)
    element.load_weights(result.weights, result.biases)
    element.history = result.history

    # Load the dataset
    try:
        X, y = get_dataset(metadata.dataset_name)
    except ValueError:
        return jsonify({'error': f'Dataset {metadata.dataset_name} not found'}), 404

    # Generate boundary data
    resolution = int(request.args.get('resolution', 30))
    boundary_data = generate_decision_boundary_data(element, resolution=resolution)

    # Build response in Training Lab format
    # Create frames from history - we'll create one frame per history entry
    frames = []
    history_len = len(result.history.get('loss', []))

    if history_len > 0:
        # We only have the final trained weights, so all frames show the same boundary
        # but with different epoch labels
        record_every = max(1, 500 // history_len)  # Estimate original record_every

        for i in range(history_len):
            frames.append({
                'epoch': i * record_every,
                'probs': boundary_data['z'],  # Final trained state
            })

    response = {
        'experiment_id': experiment_id,
        'frames': frames,
        'x': boundary_data['x'],
        'y': boundary_data['y'],
        'x_range': boundary_data['x_range'],
        'y_range': boundary_data['y_range'],
        'data_points': {
            'x': X[:, 0].tolist(),
            'y': X[:, 1].tolist(),
            'labels': y.tolist(),
        },
        'loss_history': result.history.get('loss', []),
        'accuracy_history': result.history.get('accuracy', []),
        'element': {
            'name': element.name,
            'architecture': {
                'input_dim': element.config.input_dim,
                'hidden_layers': element.config.hidden_layers,
                'output_dim': element.config.output_dim,
                'activation': element.config.activation,
            },
            'properties': {
                'depth': element.config.depth,
                'total_params': element.config.total_params,
                'width_pattern': element.config.width_pattern,
            },
            'training': {
                'final_loss': result.final_loss,
                'final_accuracy': result.final_accuracy,
            },
        },
        'dataset': metadata.dataset_name,
        'loaded_from_experiment': True,
    }

    return jsonify(response)


@bulk_bp.route('/experiments/by-element/<element_name>', methods=['GET'])
def get_experiments_by_element(element_name: str):
    """
    Get all experiments for a specific element type.

    Useful for showing experiment history in the Periodic Table element detail view.
    """
    store = get_experiment_store()

    # Get all experiments and filter by element name
    all_experiments = store.list_experiments(status='completed', limit=1000)
    matching = [e for e in all_experiments if e.get('element_name') == element_name]

    # Sort by accuracy descending
    matching.sort(key=lambda e: e.get('final_accuracy', 0), reverse=True)

    return jsonify({
        'element_name': element_name,
        'experiments': matching[:20],  # Return top 20
        'total': len(matching),
    })
