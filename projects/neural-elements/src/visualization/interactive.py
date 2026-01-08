"""
Data generation for interactive D3.js visualizations.

These functions generate JSON-serializable data structures
for use in the web frontend.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def generate_decision_boundary_data(
    element,
    x_range: Tuple[float, float] = (-3, 3),
    y_range: Tuple[float, float] = (-3, 3),
    resolution: int = 50
) -> Dict[str, Any]:
    """
    Generate decision boundary data for D3.js visualization.

    Returns a grid of probabilities for contour plotting.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = element.predict_proba(grid).reshape(xx.shape)

    return {
        'x': x.tolist(),
        'y': y.tolist(),
        'z': probs.tolist(),
        'x_range': list(x_range),
        'y_range': list(y_range),
        'resolution': resolution,
    }


def generate_training_animation_data(
    element,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 500,
    record_every: int = 10,
    resolution: int = 30
) -> Dict[str, Any]:
    """
    Generate training animation data by training and recording snapshots.

    Returns data for animating decision boundary evolution during training.
    """
    element.reset()

    x_range = (float(X[:, 0].min() - 0.5), float(X[:, 0].max() + 0.5))
    y_range = (float(X[:, 1].min() - 0.5), float(X[:, 1].max() + 0.5))

    x = np.linspace(x_range[0], x_range[1], resolution)
    y_coords = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y_coords)
    grid = np.c_[xx.ravel(), yy.ravel()]

    frames = []
    loss_history = []
    accuracy_history = []

    y_train = y.reshape(-1, 1)

    for epoch in range(epochs):
        # Training step
        weight_grads, bias_grads = element.backward(X, y)
        for i in range(len(element.weights)):
            element.weights[i] -= 0.1 * weight_grads[i]
            if element.biases[i] is not None:
                element.biases[i] -= 0.1 * bias_grads[i]

        # Record frame
        if epoch % record_every == 0 or epoch == epochs - 1:
            probs = element.predict_proba(grid).reshape(xx.shape)
            output = element.forward(X)
            loss = float(element._binary_cross_entropy(y_train, output))
            accuracy = float(np.mean((output > 0.5).astype(int) == y_train))

            frames.append({
                'epoch': epoch,
                'probs': probs.tolist(),
            })
            loss_history.append(loss)
            accuracy_history.append(accuracy)

    element.trained = True
    element.history = {'loss': loss_history, 'accuracy': accuracy_history}

    return {
        'frames': frames,
        'x': x.tolist(),
        'y': y_coords.tolist(),
        'x_range': list(x_range),
        'y_range': list(y_range),
        'data_points': {
            'x': X[:, 0].tolist(),
            'y': X[:, 1].tolist(),
            'labels': y.tolist(),
        },
        'loss_history': loss_history,
        'accuracy_history': accuracy_history,
        'total_frames': len(frames),
    }


def generate_periodic_table_data(registry) -> Dict[str, Any]:
    """
    Generate data structure for periodic table visualization.

    Organizes elements by depth (period) and activation (group).
    """
    # Standard groups (columns)
    groups = ['linear', 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'sine', 'gelu', 'swish']

    # Periods (rows) by depth
    max_depth = max((e.config.depth for e in registry), default=3)
    periods = list(range(1, max_depth + 1))

    # Build table structure
    table = {
        'groups': groups,
        'periods': periods,
        'elements': {},
        'group_info': {},
        'period_info': {},
    }

    # Group descriptions
    group_descriptions = {
        'linear': 'No nonlinearity',
        'relu': 'Piecewise linear, sparse',
        'leaky_relu': 'Leaky gradients',
        'tanh': 'Smooth, zero-centered',
        'sigmoid': 'Smooth, bounded [0,1]',
        'sine': 'Periodic activation',
        'gelu': 'Smooth, transformer-style',
        'swish': 'Self-gated activation',
    }

    for group in groups:
        table['group_info'][group] = {
            'name': group.upper(),
            'description': group_descriptions.get(group, ''),
        }

    for period in periods:
        table['period_info'][str(period)] = {
            'depth': period,
            'description': f'{period} hidden layer{"s" if period > 1 else ""}',
        }

    # Place elements in table
    for element in registry:
        key = f"{element.config.depth}-{element.config.activation}"
        # Only include standard width (4) elements in main table
        if element.config.hidden_layers and element.config.hidden_layers[0] == 4:
            table['elements'][key] = {
                'name': element.name,
                'period': element.config.depth,
                'group': element.config.activation,
                'params': element.config.total_params,
                'architecture': element.config.hidden_layers,
                'trained': element.trained,
                'properties': {
                    'depth': element.config.depth,
                    'width': element.config.hidden_layers[0] if element.config.hidden_layers else 0,
                    'activation_family': element.activation_fn.family,
                },
            }

    return table


def generate_element_card_data(element) -> Dict[str, Any]:
    """Generate detailed data for element info card."""
    data = {
        'name': element.name,
        'architecture': {
            'input_dim': element.config.input_dim,
            'hidden_layers': element.config.hidden_layers,
            'output_dim': element.config.output_dim,
            'activation': element.config.activation,
        },
        'properties': {
            'total_params': element.config.total_params,
            'depth': element.config.depth,
            'width_pattern': element.config.width_pattern,
        },
        'activation_info': {
            'name': element.activation_fn.name,
            'family': element.activation_fn.family,
            **element.activation_fn.properties,
        },
        'trained': element.trained,
    }

    if element.trained and element.history['loss']:
        data['training'] = {
            'final_loss': element.history['loss'][-1],
            'final_accuracy': element.history['accuracy'][-1],
            'epochs_trained': len(element.history['loss']),
        }

    return data


def generate_comparison_data(
    elements: List,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 30
) -> Dict[str, Any]:
    """Generate comparison data for multiple elements."""
    x_range = (float(X[:, 0].min() - 0.5), float(X[:, 0].max() + 0.5))
    y_range = (float(X[:, 1].min() - 0.5), float(X[:, 1].max() + 0.5))

    x = np.linspace(x_range[0], x_range[1], resolution)
    y_coords = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y_coords)
    grid = np.c_[xx.ravel(), yy.ravel()]

    elements_data = []

    for element in elements:
        probs = element.predict_proba(grid).reshape(xx.shape)
        elements_data.append({
            'name': element.name,
            'probs': probs.tolist(),
            'params': element.config.total_params,
            'accuracy': element.history['accuracy'][-1] if element.history['accuracy'] else None,
        })

    return {
        'x': x.tolist(),
        'y': y_coords.tolist(),
        'x_range': list(x_range),
        'y_range': list(y_range),
        'data_points': {
            'x': X[:, 0].tolist(),
            'y': X[:, 1].tolist(),
            'labels': y.tolist(),
        },
        'elements': elements_data,
    }
