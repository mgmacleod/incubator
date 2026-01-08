"""Core neural elements framework."""

from .elements import NeuralElement, ElementRegistry
from .activations import ACTIVATIONS, get_activation
from .training import Trainer
from .properties import compute_properties

__all__ = [
    'NeuralElement',
    'ElementRegistry',
    'ACTIVATIONS',
    'get_activation',
    'Trainer',
    'compute_properties',
]
