"""
Neural Elements - the fundamental building blocks of our periodic table.

A Neural Element is a minimal neural network configuration characterized by:
- Depth (number of hidden layers) - analogous to "period" in chemistry
- Activation function - analogous to "group" in chemistry
- Width pattern - analogous to "block" in chemistry
- Input/output dimensions

Elements can be trained on various tasks to discover their computational properties.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from .activations import get_activation, Activation, ACTIVATIONS


@dataclass
class ElementConfig:
    """Configuration for a neural element."""
    hidden_layers: List[int]  # Width of each hidden layer
    activation: str  # Name of activation function
    input_dim: int = 2  # Input dimension (default 2D for visualization)
    output_dim: int = 1  # Output dimension
    bias: bool = True  # Whether to use bias terms
    skip_connections: bool = False  # Whether to use residual connections

    @property
    def depth(self) -> int:
        """Number of hidden layers (period in our table)."""
        return len(self.hidden_layers)

    @property
    def total_params(self) -> int:
        """Total number of trainable parameters."""
        params = 0
        prev_dim = self.input_dim
        for width in self.hidden_layers:
            params += prev_dim * width  # weights
            if self.bias:
                params += width  # biases
            prev_dim = width
        # Output layer
        params += prev_dim * self.output_dim
        if self.bias:
            params += self.output_dim
        return params

    @property
    def width_pattern(self) -> str:
        """Categorize the width pattern."""
        if len(self.hidden_layers) == 0:
            return 'linear'
        widths = self.hidden_layers
        if all(w == widths[0] for w in widths):
            return 'uniform'
        if widths == sorted(widths):
            return 'expanding'
        if widths == sorted(widths, reverse=True):
            return 'contracting'
        return 'mixed'


class NeuralElement:
    """
    A neural element - a minimal neural network that can be trained and analyzed.

    This is the "atom" of our periodic table. Each element has:
    - A specific architecture (depth, widths, activation)
    - Learnable parameters (weights and biases)
    - Methods for forward/backward passes
    - Properties that can be measured after training
    """

    def __init__(
        self,
        hidden_layers: List[int],
        activation: str = 'relu',
        input_dim: int = 2,
        output_dim: int = 1,
        bias: bool = True,
        skip_connections: bool = False,
        name: Optional[str] = None,
        seed: Optional[int] = None
    ):
        self.config = ElementConfig(
            hidden_layers=hidden_layers,
            activation=activation,
            input_dim=input_dim,
            output_dim=output_dim,
            bias=bias,
            skip_connections=skip_connections
        )

        # Store activation function first (needed for get_element_name)
        self.activation_fn = get_activation(activation)
        self.seed = seed

        # Generate name if not provided
        if name is None:
            self.name = self.get_element_name()
        else:
            self.name = name

        # Initialize weights
        self._init_weights()

        # Training history
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
        }
        self.trained = False
        self._training_snapshots: List[Dict] = []

        # Layer freezing for transfer learning (Phase 8)
        self.frozen_layers: List[bool] = []

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        layer_dims = [self.config.input_dim] + self.config.hidden_layers + [self.config.output_dim]

        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i + 1]

            # He initialization for ReLU-like, Xavier for others
            if self.activation_fn.family == 'rectified':
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))

            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out) if self.config.bias else None

            self.weights.append(W)
            self.biases.append(b)

        # Initialize frozen_layers list after weights are created
        self.frozen_layers = [False] * len(self.weights)

    def forward(self, X: np.ndarray, return_intermediates: bool = False) -> Any:
        """
        Forward pass through the network.

        Args:
            X: Input array of shape (n_samples, input_dim)
            return_intermediates: If True, return all intermediate activations

        Returns:
            Output predictions, optionally with intermediate values
        """
        intermediates = {'pre_activations': [], 'activations': [X], 'skip_applied': []}

        current = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Store input before transformation (for skip connections)
            layer_input = current

            # Linear transformation
            z = current @ W
            if b is not None:
                z = z + b
            intermediates['pre_activations'].append(z)

            # Apply activation (except for output layer)
            if i < len(self.weights) - 1:
                activated = self.activation_fn(z)

                # Apply skip connection if enabled and dimensions match
                skip_applied = False
                if self.config.skip_connections and z.shape[1] == layer_input.shape[1]:
                    current = activated + layer_input  # Residual connection
                    skip_applied = True
                else:
                    current = activated

                intermediates['skip_applied'].append(skip_applied)
            else:
                # Output layer: sigmoid for binary classification
                current = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                intermediates['skip_applied'].append(False)

            intermediates['activations'].append(current)

        if return_intermediates:
            return current, intermediates
        return current

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions (0 or 1 for classification)."""
        probs = self.forward(X)
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        return self.forward(X)

    def backward(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass - compute gradients.

        Handles skip connections by splitting gradients:
        - Through transformation path: delta * activation_grad(z)
        - Through skip path: delta (identity)

        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        n_samples = X.shape[0]
        y = y.reshape(-1, 1)

        # Forward pass with intermediates
        output, intermediates = self.forward(X, return_intermediates=True)

        # Initialize gradients
        weight_grads = [np.zeros_like(W) for W in self.weights]
        bias_grads = [np.zeros_like(b) if b is not None else None for b in self.biases]

        # Output layer gradient (binary cross-entropy)
        delta = output - y  # Shape: (n_samples, output_dim)

        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights
            prev_activation = intermediates['activations'][i]
            weight_grads[i] = (prev_activation.T @ delta) / n_samples

            # Gradient for bias
            if bias_grads[i] is not None:
                bias_grads[i] = np.mean(delta, axis=0)

            # Propagate delta to previous layer
            if i > 0:
                # Gradient through the transformation path
                delta_transform = (delta @ self.weights[i].T) * self.activation_fn.grad(
                    intermediates['pre_activations'][i - 1]
                )

                # If skip connection was applied at this layer, add identity gradient
                if intermediates['skip_applied'][i - 1]:
                    # Skip connection: gradient flows through both paths
                    delta = delta_transform + delta
                else:
                    delta = delta_transform

        return weight_grads, bias_grads

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.1,
        batch_size: Optional[int] = None,
        record_every: int = 10,
        verbose: bool = False
    ) -> Dict[str, List[float]]:
        """
        Train the neural element on data.

        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            batch_size: Mini-batch size (None for full batch)
            record_every: Record history every N epochs
            verbose: Print training progress

        Returns:
            Training history
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples = X.shape[0]

        if batch_size is None:
            batch_size = n_samples

        self.history = {'loss': [], 'accuracy': []}
        self._training_snapshots = []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Compute gradients
                weight_grads, bias_grads = self.backward(X_batch, y_batch)

                # Update weights (skip frozen layers)
                for i in range(len(self.weights)):
                    if self.frozen_layers[i]:
                        continue  # Skip frozen layers
                    self.weights[i] -= learning_rate * weight_grads[i]
                    if self.biases[i] is not None:
                        self.biases[i] -= learning_rate * bias_grads[i]

            # Record history
            if epoch % record_every == 0 or epoch == epochs - 1:
                output = self.forward(X)
                loss = self._binary_cross_entropy(y, output)
                accuracy = np.mean((output > 0.5).astype(int) == y)

                self.history['loss'].append(loss)
                self.history['accuracy'].append(accuracy)

                # Save snapshot for visualization
                self._training_snapshots.append({
                    'epoch': epoch,
                    'loss': loss,
                    'accuracy': accuracy,
                    'weights': [W.copy() for W in self.weights],
                    'biases': [b.copy() if b is not None else None for b in self.biases]
                })

                if verbose and epoch % (record_every * 10) == 0:
                    print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")

        self.trained = True
        return self.history

    def _binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def get_decision_boundary_data(
        self,
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        resolution: int = 100
    ) -> Dict:
        """Generate data for visualizing decision boundary."""
        xx, yy = np.meshgrid(
            np.linspace(x_range[0], x_range[1], resolution),
            np.linspace(y_range[0], y_range[1], resolution)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = self.predict_proba(grid).reshape(xx.shape)

        return {
            'xx': xx.tolist(),
            'yy': yy.tolist(),
            'probs': probs.tolist()
        }

    def get_element_name(self) -> str:
        """
        Generate a descriptive name for this element.

        Handles both uniform and non-uniform architectures:
        - Uniform [8, 8, 8]: "rel-3x8" (activation-depthxwidth)
        - Non-uniform [32, 8, 32]: "rel-32_8_32" (activation-widths joined by _)
        - With skip connections: append "-skip"
        """
        activation = self.config.activation[:3].lower()
        skip_suffix = "-skip" if self.config.skip_connections else ""

        if self.config.width_pattern == 'uniform' and self.config.hidden_layers:
            # Original format for uniform architectures: rel-3x8
            width = self.config.hidden_layers[0]
            return f"{activation}-{self.config.depth}x{width}{skip_suffix}"
        elif self.config.hidden_layers:
            # New format for non-uniform: rel-32_8_32
            widths = '_'.join(str(w) for w in self.config.hidden_layers)
            return f"{activation}-{widths}{skip_suffix}"
        else:
            # Linear (no hidden layers)
            return f"{activation}-linear{skip_suffix}"

    def reset(self):
        """Reset the element to initial state."""
        self._init_weights()
        self.history = {'loss': [], 'accuracy': []}
        self._training_snapshots = []
        self.trained = False

    def copy(self) -> 'NeuralElement':
        """Create a copy of this element."""
        new_element = NeuralElement(
            hidden_layers=self.config.hidden_layers.copy(),
            activation=self.config.activation,
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            bias=self.config.bias,
            skip_connections=self.config.skip_connections,
            name=self.name,
            seed=None  # Don't copy seed to get different initialization
        )
        return new_element

    def to_dict(self, include_weights: bool = False) -> Dict:
        """
        Serialize element to dictionary.

        Args:
            include_weights: If True, include weight matrices for persistence.
        """
        data = {
            'name': self.name,
            'config': {
                'hidden_layers': self.config.hidden_layers,
                'activation': self.config.activation,
                'input_dim': self.config.input_dim,
                'output_dim': self.config.output_dim,
                'bias': self.config.bias,
                'skip_connections': self.config.skip_connections,
            },
            'properties': {
                'depth': self.config.depth,
                'total_params': self.config.total_params,
                'width_pattern': self.config.width_pattern,
                'activation_family': self.activation_fn.family,
            },
            'trained': self.trained,
            'history': self.history,
        }

        if include_weights:
            data['weights'] = [w.tolist() for w in self.weights]
            data['biases'] = [b.tolist() if b is not None else None for b in self.biases]

        return data

    def load_weights(self, weights: List[List], biases: List[List]):
        """
        Load pre-trained weights into the element.

        Args:
            weights: List of weight matrices as nested lists.
            biases: List of bias vectors as lists.
        """
        if len(weights) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} weight matrices, got {len(weights)}")

        self.weights = [np.array(w) for w in weights]
        self.biases = [np.array(b) if b is not None else None for b in biases]
        self.trained = True

    # Layer freezing methods for Phase 8 transfer learning

    def freeze_layer(self, layer_idx: int) -> None:
        """
        Freeze a layer to prevent weight updates during training.

        Args:
            layer_idx: Index of the layer to freeze (0-indexed).
        """
        if layer_idx < 0 or layer_idx >= len(self.weights):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {len(self.weights) - 1}]")
        self.frozen_layers[layer_idx] = True

    def unfreeze_layer(self, layer_idx: int) -> None:
        """
        Unfreeze a layer to allow weight updates during training.

        Args:
            layer_idx: Index of the layer to unfreeze (0-indexed).
        """
        if layer_idx < 0 or layer_idx >= len(self.weights):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {len(self.weights) - 1}]")
        self.frozen_layers[layer_idx] = False

    def freeze_all(self) -> None:
        """Freeze all layers to prevent any weight updates."""
        self.frozen_layers = [True] * len(self.weights)

    def unfreeze_all(self) -> None:
        """Unfreeze all layers to allow weight updates."""
        self.frozen_layers = [False] * len(self.weights)

    def freeze_bottom_k(self, k: int) -> None:
        """
        Freeze the bottom k layers (closest to input).

        Args:
            k: Number of layers to freeze from the input side.
        """
        for i in range(min(k, len(self.weights))):
            self.frozen_layers[i] = True

    def get_layer_output(self, X: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Get the output of a specific layer (for stacking).

        Args:
            X: Input data.
            layer_idx: Index of the layer to get output from (0-indexed).
                      Returns the activated output after that layer.

        Returns:
            Layer output as numpy array.
        """
        if layer_idx < 0 or layer_idx >= len(self.weights):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {len(self.weights) - 1}]")

        _, intermediates = self.forward(X, return_intermediates=True)
        # intermediates['activations'] has input at [0], then each layer's output
        # So layer_idx=0 output is at activations[1]
        return intermediates['activations'][layer_idx + 1]

    def n_frozen_layers(self) -> int:
        """Return the number of frozen layers."""
        return sum(self.frozen_layers)

    def is_fully_frozen(self) -> bool:
        """Return True if all layers are frozen."""
        return all(self.frozen_layers)

    @classmethod
    def from_dict(cls, data: Dict, load_weights: bool = True) -> 'NeuralElement':
        """
        Create element from dictionary.

        Args:
            data: Serialized element dictionary.
            load_weights: If True and weights are present, load them.
        """
        config = data['config']
        element = cls(
            hidden_layers=config['hidden_layers'],
            activation=config['activation'],
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            bias=config['bias'],
            skip_connections=config.get('skip_connections', False),
            name=data['name']
        )

        if load_weights and 'weights' in data and data['weights']:
            element.load_weights(data['weights'], data['biases'])
            element.history = data.get('history', {'loss': [], 'accuracy': []})
            element.trained = data.get('trained', True)

        return element

    def __repr__(self):
        arch = f"{self.config.input_dim}→" + "→".join(map(str, self.config.hidden_layers)) + f"→{self.config.output_dim}"
        return f"NeuralElement({self.name}, arch={arch}, activation={self.config.activation}, params={self.config.total_params})"


class ElementRegistry:
    """
    Registry of neural elements organized like a periodic table.

    Elements are organized by:
    - Period (row): Depth of the network
    - Group (column): Activation function family
    - Block: Width pattern
    """

    # Standard elements in the periodic table
    STANDARD_ELEMENTS = [
        # Period 1 (1 hidden layer)
        {'hidden_layers': [4], 'activation': 'linear'},
        {'hidden_layers': [4], 'activation': 'relu'},
        {'hidden_layers': [4], 'activation': 'tanh'},
        {'hidden_layers': [4], 'activation': 'sigmoid'},
        {'hidden_layers': [4], 'activation': 'sine'},
        {'hidden_layers': [4], 'activation': 'gelu'},

        # Period 2 (2 hidden layers)
        {'hidden_layers': [4, 4], 'activation': 'linear'},
        {'hidden_layers': [4, 4], 'activation': 'relu'},
        {'hidden_layers': [4, 4], 'activation': 'tanh'},
        {'hidden_layers': [4, 4], 'activation': 'sigmoid'},
        {'hidden_layers': [4, 4], 'activation': 'sine'},
        {'hidden_layers': [4, 4], 'activation': 'gelu'},

        # Period 3 (3 hidden layers)
        {'hidden_layers': [4, 4, 4], 'activation': 'linear'},
        {'hidden_layers': [4, 4, 4], 'activation': 'relu'},
        {'hidden_layers': [4, 4, 4], 'activation': 'tanh'},
        {'hidden_layers': [4, 4, 4], 'activation': 'sigmoid'},
        {'hidden_layers': [4, 4, 4], 'activation': 'sine'},
        {'hidden_layers': [4, 4, 4], 'activation': 'gelu'},

        # Wider variants (Block W)
        {'hidden_layers': [8], 'activation': 'relu'},
        {'hidden_layers': [8, 8], 'activation': 'relu'},
        {'hidden_layers': [16], 'activation': 'relu'},
        {'hidden_layers': [16, 16], 'activation': 'relu'},

        # Narrow variants (Block N)
        {'hidden_layers': [2], 'activation': 'relu'},
        {'hidden_layers': [2, 2], 'activation': 'relu'},
        {'hidden_layers': [3], 'activation': 'tanh'},
        {'hidden_layers': [3, 3], 'activation': 'tanh'},
    ]

    def __init__(self):
        self.elements: Dict[str, NeuralElement] = {}
        self._load_standard_elements()

    def _load_standard_elements(self):
        """Load the standard set of elements."""
        for config in self.STANDARD_ELEMENTS:
            element = NeuralElement(**config)
            self.register(element)

    def register(self, element: NeuralElement):
        """Register an element in the registry."""
        self.elements[element.name] = element

    def get(self, name: str) -> Optional[NeuralElement]:
        """Get an element by name."""
        return self.elements.get(name)

    def list_by_period(self, period: int) -> List[NeuralElement]:
        """Get all elements with a specific depth (period)."""
        return [e for e in self.elements.values() if e.config.depth == period]

    def list_by_activation(self, activation: str) -> List[NeuralElement]:
        """Get all elements with a specific activation."""
        return [e for e in self.elements.values() if e.config.activation == activation]

    def get_table_data(self) -> Dict:
        """
        Get data organized as a periodic table.

        Returns dict with structure:
        {
            'periods': [1, 2, 3],
            'groups': ['linear', 'relu', 'tanh', ...],
            'elements': {
                (period, group): element_data
            }
        }
        """
        periods = sorted(set(e.config.depth for e in self.elements.values()))
        groups = ['linear', 'relu', 'tanh', 'sigmoid', 'sine', 'gelu']

        table = {
            'periods': periods,
            'groups': groups,
            'elements': {}
        }

        for element in self.elements.values():
            key = f"{element.config.depth}-{element.config.activation}"
            if element.config.hidden_layers and element.config.hidden_layers[0] == 4:
                # Only include standard width in main table
                table['elements'][key] = element.to_dict()

        return table

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements.values())


# Convenience function to create common elements
def create_element(
    depth: int = 2,
    width: int = 4,
    activation: str = 'relu',
    **kwargs
) -> NeuralElement:
    """Create a neural element with uniform width."""
    hidden_layers = [width] * depth
    return NeuralElement(hidden_layers=hidden_layers, activation=activation, **kwargs)
