"""
Dynamics analysis for Phase 6: Learning Dynamics Study.

Computes derived metrics from training history to understand
how different elements learn, not just their final performance.
"""

import numpy as np
from typing import Dict, List, Any, Optional


def compute_dynamics_metrics(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute derived dynamics metrics from training history.

    Args:
        history: Training history dict with keys like 'loss', 'accuracy',
                 and optionally 'gradient_stats', 'weight_stats'

    Returns:
        Dictionary of derived metrics
    """
    metrics = {}

    # 1. Convergence speed: epochs to reach 90% of final accuracy
    if 'accuracy' in history and len(history['accuracy']) > 0:
        final_acc = history['accuracy'][-1]
        target_acc = 0.9 * final_acc
        convergence_idx = next(
            (i for i, acc in enumerate(history['accuracy']) if acc >= target_acc),
            len(history['accuracy']) - 1
        )
        # Convert index to actual epoch (accounting for record_every)
        if 'gradient_stats' in history and len(history['gradient_stats']) > 0:
            # Use epoch from gradient stats if available
            metrics['convergence_epoch'] = history['gradient_stats'][convergence_idx].get(
                'epoch', convergence_idx * 10
            )
        else:
            metrics['convergence_epoch'] = convergence_idx * 10  # Assume record_every=10
        metrics['convergence_index'] = convergence_idx

    # 2. Training stability: std of loss in last 20% of training
    if 'loss' in history and len(history['loss']) > 0:
        n = len(history['loss'])
        tail_start = int(0.8 * n)
        if tail_start < n:
            metrics['loss_stability'] = float(np.std(history['loss'][tail_start:]))
            metrics['loss_final'] = float(history['loss'][-1])
            metrics['loss_initial'] = float(history['loss'][0])
            metrics['loss_improvement'] = float(history['loss'][0] - history['loss'][-1])

    # 3. Gradient flow health (if recorded)
    if 'gradient_stats' in history and len(history['gradient_stats']) > 0:
        metrics['gradient_health'] = compute_gradient_health(history['gradient_stats'])

    # 4. Weight evolution (if recorded)
    if 'weight_stats' in history and len(history['weight_stats']) > 0:
        metrics['weight_evolution'] = compute_weight_evolution(history['weight_stats'])

    # 5. Learning curve characteristics
    if 'accuracy' in history and len(history['accuracy']) > 1:
        metrics['learning_curve'] = compute_learning_curve_stats(history['accuracy'])

    return metrics


def compute_gradient_health(gradient_stats: List[Dict]) -> Dict[str, Any]:
    """
    Analyze gradient flow health over training.

    Args:
        gradient_stats: List of gradient stat dicts from training

    Returns:
        Dictionary with gradient health indicators
    """
    if not gradient_stats:
        return {}

    mean_grads = [s['mean_grad'] for s in gradient_stats]
    max_grads = [s['max_grad'] for s in gradient_stats]

    # Filter out NaN/Inf values for robust statistics
    valid_mean_grads = [g for g in mean_grads if np.isfinite(g)]
    valid_max_grads = [g for g in max_grads if np.isfinite(g)]

    if not valid_mean_grads:
        # All gradients were NaN/Inf - severe gradient issues
        return {
            'mean_grad_overall': float('nan'),
            'mean_grad_early': float('nan'),
            'mean_grad_late': float('nan'),
            'mean_grad_final': float('nan'),
            'max_grad_overall': float('nan'),
            'grad_trend_slope': 0.0,
            'vanishing': False,
            'exploding': True,  # Assume exploding if all NaN
            'gradient_ratio': 0.0,
            'layer_health': [],
        }

    # Gradient at different stages (using valid grads only)
    early_grad = np.mean(valid_mean_grads[:len(valid_mean_grads)//4]) if len(valid_mean_grads) >= 4 else valid_mean_grads[0]
    late_grad = np.mean(valid_mean_grads[-len(valid_mean_grads)//4:]) if len(valid_mean_grads) >= 4 else valid_mean_grads[-1]

    # Trend: fit linear regression (only if all values are finite)
    if len(valid_mean_grads) > 1 and len(valid_mean_grads) == len(mean_grads):
        try:
            slope, intercept = np.polyfit(range(len(mean_grads)), mean_grads, 1)
        except (np.linalg.LinAlgError, ValueError):
            slope, intercept = 0.0, valid_mean_grads[0]
    else:
        slope, intercept = 0.0, valid_mean_grads[0] if valid_mean_grads else 0.0

    # Per-layer analysis
    n_layers = len(gradient_stats[0].get('layer_grad_norms', []))
    layer_health = []
    if n_layers > 0:
        for layer_idx in range(n_layers):
            layer_grads = [s['layer_grad_norms'][layer_idx] for s in gradient_stats]
            valid_layer_grads = [g for g in layer_grads if np.isfinite(g)]
            if valid_layer_grads:
                try:
                    trend = float(np.polyfit(range(len(valid_layer_grads)), valid_layer_grads, 1)[0]) if len(valid_layer_grads) > 1 else 0.0
                except (np.linalg.LinAlgError, ValueError):
                    trend = 0.0
                layer_health.append({
                    'layer': layer_idx,
                    'mean': float(np.mean(valid_layer_grads)),
                    'std': float(np.std(valid_layer_grads)),
                    'final': float(valid_layer_grads[-1]),
                    'trend': trend,
                })

    # Get final values safely
    final_mean_grad = mean_grads[-1] if np.isfinite(mean_grads[-1]) else 0.0
    final_max_grad = max_grads[-1] if np.isfinite(max_grads[-1]) else float('inf')

    return {
        'mean_grad_overall': float(np.mean(valid_mean_grads)),
        'mean_grad_early': float(early_grad),
        'mean_grad_late': float(late_grad),
        'mean_grad_final': float(final_mean_grad),
        'max_grad_overall': float(np.max(valid_max_grads)) if valid_max_grads else float('nan'),
        'grad_trend_slope': float(slope),
        'vanishing': final_mean_grad < 1e-6 if np.isfinite(final_mean_grad) else False,
        'exploding': final_max_grad > 100 or not np.isfinite(final_max_grad),
        'gradient_ratio': float(late_grad / early_grad) if early_grad > 1e-10 else 0.0,
        'layer_health': layer_health,
    }


def compute_weight_evolution(weight_stats: List[Dict]) -> Dict[str, Any]:
    """
    Analyze weight distribution evolution over training.

    Args:
        weight_stats: List of weight stat dicts from training

    Returns:
        Dictionary with weight evolution metrics
    """
    if not weight_stats:
        return {}

    mean_weights = [s['mean_weight'] for s in weight_stats]
    std_weights = [s['std_weight'] for s in weight_stats]

    # Weight norms over time
    n_layers = len(weight_stats[0].get('layer_weight_norms', []))
    layer_norms_evolution = []
    if n_layers > 0:
        for layer_idx in range(n_layers):
            layer_norms = [s['layer_weight_norms'][layer_idx] for s in weight_stats]
            layer_norms_evolution.append({
                'layer': layer_idx,
                'initial': float(layer_norms[0]),
                'final': float(layer_norms[-1]),
                'change': float(layer_norms[-1] - layer_norms[0]),
                'change_pct': float((layer_norms[-1] - layer_norms[0]) / layer_norms[0] * 100) if layer_norms[0] > 1e-10 else 0.0,
            })

    return {
        'mean_weight_initial': float(mean_weights[0]),
        'mean_weight_final': float(mean_weights[-1]),
        'std_weight_initial': float(std_weights[0]),
        'std_weight_final': float(std_weights[-1]),
        'weight_growth': float(std_weights[-1] / std_weights[0]) if std_weights[0] > 1e-10 else 1.0,
        'layer_evolution': layer_norms_evolution,
    }


def compute_learning_curve_stats(accuracy: List[float]) -> Dict[str, Any]:
    """
    Analyze the shape of the learning curve.

    Args:
        accuracy: List of accuracy values over training

    Returns:
        Dictionary with learning curve characteristics
    """
    if len(accuracy) < 2:
        return {}

    accuracy = np.array(accuracy)

    # Find plateau regions (where improvement is minimal)
    diffs = np.diff(accuracy)
    plateau_threshold = 0.001  # 0.1% improvement threshold

    # Early learning rate (first 25% of training)
    early_end = len(accuracy) // 4
    early_improvement = accuracy[early_end] - accuracy[0] if early_end > 0 else 0

    # Late learning (last 25% of training)
    late_start = 3 * len(accuracy) // 4
    late_improvement = accuracy[-1] - accuracy[late_start] if late_start < len(accuracy) else 0

    # Monotonicity: how often does accuracy increase?
    increasing_steps = np.sum(diffs > 0)
    monotonicity = increasing_steps / len(diffs) if len(diffs) > 0 else 1.0

    return {
        'initial_accuracy': float(accuracy[0]),
        'final_accuracy': float(accuracy[-1]),
        'total_improvement': float(accuracy[-1] - accuracy[0]),
        'early_improvement': float(early_improvement),
        'late_improvement': float(late_improvement),
        'monotonicity': float(monotonicity),
        'max_accuracy': float(np.max(accuracy)),
        'min_accuracy': float(np.min(accuracy)),
    }


def classify_training_pattern(metrics: Dict[str, Any]) -> str:
    """
    Classify the training pattern based on dynamics metrics.

    Args:
        metrics: Output from compute_dynamics_metrics

    Returns:
        String describing the training pattern
    """
    patterns = []

    # Check for vanishing gradients
    if metrics.get('gradient_health', {}).get('vanishing', False):
        patterns.append('vanishing_gradients')

    # Check for exploding gradients
    if metrics.get('gradient_health', {}).get('exploding', False):
        patterns.append('exploding_gradients')

    # Check for slow convergence
    convergence = metrics.get('convergence_epoch', 0)
    if convergence > 500:
        patterns.append('slow_convergence')
    elif convergence < 100:
        patterns.append('fast_convergence')

    # Check for unstable training
    stability = metrics.get('loss_stability', 0)
    if stability > 0.1:
        patterns.append('unstable')
    elif stability < 0.01:
        patterns.append('stable')

    # Check for gradient decay
    grad_ratio = metrics.get('gradient_health', {}).get('gradient_ratio', 1.0)
    if grad_ratio < 0.1:
        patterns.append('gradient_decay')
    elif grad_ratio > 10:
        patterns.append('gradient_growth')

    if not patterns:
        patterns.append('normal')

    return ','.join(patterns)


def aggregate_dynamics_across_trials(
    trial_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate dynamics metrics across multiple trials.

    Args:
        trial_metrics: List of metrics dicts, one per trial

    Returns:
        Aggregated metrics with mean, std, etc.
    """
    if not trial_metrics:
        return {}

    aggregated = {}

    # Scalar metrics to aggregate
    scalar_keys = [
        'convergence_epoch',
        'loss_stability',
        'loss_final',
        'loss_improvement',
    ]

    for key in scalar_keys:
        values = [m.get(key) for m in trial_metrics if m.get(key) is not None]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))

    # Gradient health aggregation
    grad_health_keys = [
        'mean_grad_overall',
        'mean_grad_final',
        'grad_trend_slope',
        'gradient_ratio',
    ]

    for key in grad_health_keys:
        values = [
            m.get('gradient_health', {}).get(key)
            for m in trial_metrics
            if m.get('gradient_health', {}).get(key) is not None
        ]
        if values:
            aggregated[f'grad_{key}_mean'] = float(np.mean(values))
            aggregated[f'grad_{key}_std'] = float(np.std(values))

    # Count training patterns
    patterns = [classify_training_pattern(m) for m in trial_metrics]
    pattern_counts = {}
    for p in patterns:
        for pattern in p.split(','):
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    aggregated['pattern_counts'] = pattern_counts

    # Vanishing gradient rate
    vanishing_count = sum(
        1 for m in trial_metrics
        if m.get('gradient_health', {}).get('vanishing', False)
    )
    aggregated['vanishing_gradient_rate'] = vanishing_count / len(trial_metrics)

    return aggregated
