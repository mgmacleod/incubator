"""
Property calculations for neural elements.

These properties help characterize elements and organize them in the periodic table.
Properties include:
- Structural: depth, width, parameter count
- Capacity: theoretical expressiveness
- Empirical: measured from training
- Behavioral: how the element learns
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .elements import NeuralElement


def compute_properties(element: NeuralElement) -> Dict:
    """
    Compute all properties for a neural element.

    Returns a dictionary of properties organized by category.
    """
    properties = {
        'structural': compute_structural_properties(element),
        'capacity': compute_capacity_properties(element),
        'activation': compute_activation_properties(element),
    }

    if element.trained:
        properties['empirical'] = compute_empirical_properties(element)

    return properties


def compute_structural_properties(element: NeuralElement) -> Dict:
    """Compute properties derived from network structure."""
    config = element.config

    # Layer statistics
    all_widths = [config.input_dim] + config.hidden_layers + [config.output_dim]

    return {
        'depth': config.depth,
        'total_layers': len(all_widths),
        'hidden_layers': config.depth,
        'total_params': config.total_params,
        'input_dim': config.input_dim,
        'output_dim': config.output_dim,
        'max_width': max(config.hidden_layers) if config.hidden_layers else 0,
        'min_width': min(config.hidden_layers) if config.hidden_layers else 0,
        'avg_width': np.mean(config.hidden_layers) if config.hidden_layers else 0,
        'width_pattern': config.width_pattern,
        'bottleneck_ratio': (min(config.hidden_layers) / max(config.hidden_layers)
                           if config.hidden_layers and max(config.hidden_layers) > 0 else 1.0),
        'has_bias': config.bias,
    }


def compute_capacity_properties(element: NeuralElement) -> Dict:
    """
    Compute theoretical capacity properties.

    These measure the potential expressiveness of the architecture.
    """
    config = element.config

    # Number of linear regions (for ReLU-like networks)
    # Rough upper bound: product of widths for piecewise linear activations
    if element.activation_fn.family == 'rectified':
        # Upper bound on linear regions
        linear_regions = 1
        for w in config.hidden_layers:
            linear_regions *= (2 ** w)
    else:
        linear_regions = float('inf')  # Smooth activations have infinite regions

    # VC dimension approximation (very rough)
    # For neural networks: O(params * depth * log(params))
    params = config.total_params
    depth = max(1, config.depth)
    vc_dimension_approx = params * depth * np.log(max(2, params))

    # Memorization capacity
    # A network with N parameters can memorize ~N/2 points
    memorization_capacity = config.total_params // 2

    return {
        'linear_regions': linear_regions if linear_regions < 1e10 else 'unbounded',
        'vc_dimension_approx': float(vc_dimension_approx),
        'memorization_capacity': memorization_capacity,
        'params_per_layer': config.total_params / max(1, config.depth),
        'depth_to_width_ratio': config.depth / max(1, np.mean(config.hidden_layers)) if config.hidden_layers else 0,
    }


def compute_activation_properties(element: NeuralElement) -> Dict:
    """Properties related to the activation function."""
    act = element.activation_fn

    return {
        'name': act.name,
        'family': act.family,
        'bounded': act.properties.get('bounded', False),
        'smooth': act.properties.get('smooth', True),
        'sparse': act.properties.get('sparse', False),
        'monotonic': act.properties.get('monotonic', True),
        'range': act.properties.get('range', (-np.inf, np.inf)),
    }


def compute_empirical_properties(element: NeuralElement) -> Dict:
    """
    Compute properties measured from training.

    Requires the element to have been trained.
    """
    if not element.trained or not element.history['loss']:
        return {}

    loss_history = element.history['loss']
    acc_history = element.history['accuracy']

    # Convergence metrics
    final_loss = loss_history[-1]
    final_accuracy = acc_history[-1]
    initial_loss = loss_history[0]

    # Learning dynamics
    loss_array = np.array(loss_history)
    loss_diff = np.diff(loss_array)

    # Find convergence point (when loss stops decreasing significantly)
    threshold = 0.001
    converged_idx = len(loss_history)
    for i, diff in enumerate(loss_diff):
        if abs(diff) < threshold:
            converged_idx = i + 1
            break

    # Smoothness of learning (variance in loss differences)
    learning_smoothness = 1.0 / (1.0 + np.std(loss_diff)) if len(loss_diff) > 0 else 1.0

    return {
        'final_loss': float(final_loss),
        'final_accuracy': float(final_accuracy),
        'initial_loss': float(initial_loss),
        'loss_reduction': float(initial_loss - final_loss),
        'loss_reduction_ratio': float((initial_loss - final_loss) / initial_loss) if initial_loss > 0 else 0,
        'epochs_to_converge': converged_idx,
        'learning_smoothness': float(learning_smoothness),
        'training_epochs': len(loss_history),
        'converged': final_loss < 0.1,
    }


def compute_weight_statistics(element: NeuralElement) -> Dict:
    """Compute statistics about the weight matrices."""
    stats = {
        'layers': [],
        'total': {}
    }

    all_weights = []

    for i, W in enumerate(element.weights):
        layer_stats = {
            'layer': i,
            'shape': W.shape,
            'mean': float(np.mean(W)),
            'std': float(np.std(W)),
            'min': float(np.min(W)),
            'max': float(np.max(W)),
            'sparsity': float(np.mean(np.abs(W) < 0.01)),
            'frobenius_norm': float(np.linalg.norm(W, 'fro')),
        }
        stats['layers'].append(layer_stats)
        all_weights.extend(W.flatten())

    all_weights = np.array(all_weights)
    stats['total'] = {
        'mean': float(np.mean(all_weights)),
        'std': float(np.std(all_weights)),
        'min': float(np.min(all_weights)),
        'max': float(np.max(all_weights)),
        'sparsity': float(np.mean(np.abs(all_weights) < 0.01)),
    }

    return stats


def measure_decision_complexity(
    element: NeuralElement,
    x_range: Tuple[float, float] = (-3, 3),
    y_range: Tuple[float, float] = (-3, 3),
    resolution: int = 100
) -> Dict:
    """
    Measure the complexity of the learned decision boundary.

    Higher complexity indicates more intricate decision boundaries.
    """
    # Generate grid
    xx, yy = np.meshgrid(
        np.linspace(x_range[0], x_range[1], resolution),
        np.linspace(y_range[0], y_range[1], resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions
    predictions = element.predict(grid).reshape(xx.shape)

    # Compute boundary complexity metrics

    # 1. Gradient magnitude at boundary (edge detection)
    dx = np.abs(np.diff(predictions, axis=1))
    dy = np.abs(np.diff(predictions, axis=0))
    boundary_length = np.sum(dx) + np.sum(dy)

    # 2. Number of connected components (rough estimate)
    # This is approximate - we count transitions
    transitions = np.sum(dx) + np.sum(dy)

    # 3. Class balance
    class_1_ratio = np.mean(predictions)

    return {
        'boundary_length': float(boundary_length),
        'transitions': int(transitions),
        'boundary_complexity': float(boundary_length / resolution),
        'class_balance': float(min(class_1_ratio, 1 - class_1_ratio) * 2),  # 1.0 = perfectly balanced
    }


def compare_element_properties(elements: List[NeuralElement]) -> Dict:
    """
    Compare properties across multiple elements.

    Useful for finding patterns in the periodic table.
    """
    comparison = {
        'elements': [],
        'summary': {}
    }

    for element in elements:
        props = compute_properties(element)
        comparison['elements'].append({
            'name': element.name,
            'properties': props
        })

    # Compute summary statistics
    if elements:
        param_counts = [e.config.total_params for e in elements]
        depths = [e.config.depth for e in elements]

        comparison['summary'] = {
            'count': len(elements),
            'param_range': (min(param_counts), max(param_counts)),
            'depth_range': (min(depths), max(depths)),
            'activations': list(set(e.config.activation for e in elements)),
        }

    return comparison
