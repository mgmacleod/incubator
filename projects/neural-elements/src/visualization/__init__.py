"""Visualization utilities for neural elements."""

from .plots import (
    plot_decision_boundary,
    plot_training_history,
    plot_activation_functions,
    plot_weight_distribution,
    plot_dataset,
)
from .interactive import (
    generate_decision_boundary_data,
    generate_training_animation_data,
    generate_periodic_table_data,
)

__all__ = [
    'plot_decision_boundary',
    'plot_training_history',
    'plot_activation_functions',
    'plot_weight_distribution',
    'plot_dataset',
    'generate_decision_boundary_data',
    'generate_training_animation_data',
    'generate_periodic_table_data',
]
