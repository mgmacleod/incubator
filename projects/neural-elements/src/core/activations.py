"""
Activation functions - the "columns" of our periodic table.

Each activation function family has distinct properties:
- Linear: No nonlinearity, purely linear transformations
- ReLU family: Piecewise linear, sparse activations
- Smooth: Differentiable everywhere, bounded or unbounded
- Periodic: Oscillating, can learn periodic patterns
"""

import numpy as np
from typing import Callable, Dict, Tuple


def linear(x: np.ndarray) -> np.ndarray:
    """Identity activation - no nonlinearity."""
    return x


def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit - most common activation."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU - allows small gradient for negative inputs."""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid - smooth, bounded (0, 1)."""
    # Clip to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent - smooth, bounded (-1, 1)."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def sine(x: np.ndarray) -> np.ndarray:
    """Sinusoidal activation - periodic, useful for coordinate networks."""
    return np.sin(x)


def sine_derivative(x: np.ndarray) -> np.ndarray:
    return np.cos(x)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit - smooth approximation of ReLU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    # Approximate derivative
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf


def swish(x: np.ndarray) -> np.ndarray:
    """Swish/SiLU - x * sigmoid(x), smooth and unbounded."""
    return x * sigmoid(x)


def swish_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s + x * s * (1 - s)


def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus - smooth approximation of ReLU."""
    return np.log1p(np.exp(np.clip(x, -500, 500)))


def softplus_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit - smooth for negative inputs."""
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))


def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha * np.exp(np.clip(x, -500, 500)))


class Activation:
    """Wrapper for activation function with its derivative and metadata."""

    def __init__(
        self,
        name: str,
        func: Callable,
        derivative: Callable,
        family: str,
        properties: Dict
    ):
        self.name = name
        self.func = func
        self.derivative = derivative
        self.family = family
        self.properties = properties

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.func(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.derivative(x)

    def __repr__(self):
        return f"Activation({self.name}, family={self.family})"


# Registry of all activation functions
ACTIVATIONS: Dict[str, Activation] = {
    'linear': Activation(
        name='linear',
        func=linear,
        derivative=linear_derivative,
        family='linear',
        properties={
            'bounded': False,
            'smooth': True,
            'sparse': False,
            'monotonic': True,
            'range': (-np.inf, np.inf),
            'description': 'Identity function - no nonlinearity'
        }
    ),
    'relu': Activation(
        name='relu',
        func=relu,
        derivative=relu_derivative,
        family='rectified',
        properties={
            'bounded': False,
            'smooth': False,
            'sparse': True,
            'monotonic': True,
            'range': (0, np.inf),
            'description': 'Rectified Linear Unit - piecewise linear, sparse'
        }
    ),
    'leaky_relu': Activation(
        name='leaky_relu',
        func=leaky_relu,
        derivative=leaky_relu_derivative,
        family='rectified',
        properties={
            'bounded': False,
            'smooth': False,
            'sparse': False,
            'monotonic': True,
            'range': (-np.inf, np.inf),
            'description': 'Leaky ReLU - allows gradient flow for negatives'
        }
    ),
    'sigmoid': Activation(
        name='sigmoid',
        func=sigmoid,
        derivative=sigmoid_derivative,
        family='smooth',
        properties={
            'bounded': True,
            'smooth': True,
            'sparse': False,
            'monotonic': True,
            'range': (0, 1),
            'description': 'Sigmoid - smooth, bounded between 0 and 1'
        }
    ),
    'tanh': Activation(
        name='tanh',
        func=tanh,
        derivative=tanh_derivative,
        family='smooth',
        properties={
            'bounded': True,
            'smooth': True,
            'sparse': False,
            'monotonic': True,
            'range': (-1, 1),
            'description': 'Hyperbolic tangent - smooth, zero-centered'
        }
    ),
    'sine': Activation(
        name='sine',
        func=sine,
        derivative=sine_derivative,
        family='periodic',
        properties={
            'bounded': True,
            'smooth': True,
            'sparse': False,
            'monotonic': False,
            'range': (-1, 1),
            'description': 'Sinusoidal - periodic, good for coordinate networks'
        }
    ),
    'gelu': Activation(
        name='gelu',
        func=gelu,
        derivative=gelu_derivative,
        family='smooth',
        properties={
            'bounded': False,
            'smooth': True,
            'sparse': False,
            'monotonic': True,
            'range': (-0.17, np.inf),
            'description': 'GELU - smooth approximation used in transformers'
        }
    ),
    'swish': Activation(
        name='swish',
        func=swish,
        derivative=swish_derivative,
        family='smooth',
        properties={
            'bounded': False,
            'smooth': True,
            'sparse': False,
            'monotonic': False,
            'range': (-0.28, np.inf),
            'description': 'Swish/SiLU - x * sigmoid(x), self-gated'
        }
    ),
    'softplus': Activation(
        name='softplus',
        func=softplus,
        derivative=softplus_derivative,
        family='smooth',
        properties={
            'bounded': False,
            'smooth': True,
            'sparse': False,
            'monotonic': True,
            'range': (0, np.inf),
            'description': 'Softplus - smooth approximation of ReLU'
        }
    ),
    'elu': Activation(
        name='elu',
        func=elu,
        derivative=elu_derivative,
        family='smooth',
        properties={
            'bounded': False,
            'smooth': True,
            'sparse': False,
            'monotonic': True,
            'range': (-1, np.inf),
            'description': 'ELU - exponential for negatives, linear for positives'
        }
    ),
}


def get_activation(name: str) -> Activation:
    """Get an activation function by name."""
    if name not in ACTIVATIONS:
        available = ', '.join(ACTIVATIONS.keys())
        raise ValueError(f"Unknown activation '{name}'. Available: {available}")
    return ACTIVATIONS[name]


def list_activations() -> Dict[str, Dict]:
    """List all available activations with their properties."""
    return {
        name: {
            'family': act.family,
            **act.properties
        }
        for name, act in ACTIVATIONS.items()
    }


# Activation families for periodic table organization
ACTIVATION_FAMILIES = {
    'linear': ['linear'],
    'rectified': ['relu', 'leaky_relu'],
    'smooth': ['sigmoid', 'tanh', 'gelu', 'swish', 'softplus', 'elu'],
    'periodic': ['sine'],
}


def get_family_activations(family: str) -> list:
    """Get all activations in a family."""
    return ACTIVATION_FAMILIES.get(family, [])
