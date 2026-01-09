"""
Training utilities for neural elements.

Provides various training algorithms and utilities for experimenting
with how different elements learn.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from .elements import NeuralElement


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 1000
    learning_rate: float = 0.1
    batch_size: Optional[int] = None
    momentum: float = 0.0
    weight_decay: float = 0.0
    lr_schedule: Optional[str] = None  # 'cosine', 'step', None
    early_stopping_patience: Optional[int] = None
    record_every: int = 10
    record_gradients: bool = False  # Record gradient statistics per layer
    record_weight_stats: bool = False  # Record weight distribution statistics


class Trainer:
    """
    Advanced trainer for neural elements.

    Supports:
    - Various optimizers (SGD, SGD with momentum)
    - Learning rate schedules
    - Early stopping
    - Detailed training logs
    """

    def __init__(self, element: NeuralElement, config: Optional[TrainingConfig] = None):
        self.element = element
        self.config = config or TrainingConfig()
        self.history: Dict[str, List] = {}
        self._velocities: Optional[List[np.ndarray]] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callbacks: Optional[List[Callable]] = None,
        verbose: bool = False
    ) -> Dict[str, List]:
        """
        Train the element.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            callbacks: List of callback functions called each epoch
            verbose: Print training progress

        Returns:
            Training history
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples = X.shape[0]

        batch_size = self.config.batch_size or n_samples
        callbacks = callbacks or []

        # Initialize history
        self.history = {
            'loss': [],
            'accuracy': [],
            'learning_rate': [],
        }
        if X_val is not None:
            self.history['val_loss'] = []
            self.history['val_accuracy'] = []
        if self.config.record_gradients:
            self.history['gradient_stats'] = []
        if self.config.record_weight_stats:
            self.history['weight_stats'] = []

        # Initialize momentum velocities
        if self.config.momentum > 0:
            self._velocities = [np.zeros_like(W) for W in self.element.weights]
            self._bias_velocities = [
                np.zeros_like(b) if b is not None else None
                for b in self.element.biases
            ]

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Get current learning rate
            lr = self._get_learning_rate(epoch)

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Compute gradients
                weight_grads, bias_grads = self.element.backward(X_batch, y_batch)

                # Apply weight decay
                if self.config.weight_decay > 0:
                    for i, W in enumerate(self.element.weights):
                        weight_grads[i] += self.config.weight_decay * W

                # Update with momentum
                if self.config.momentum > 0:
                    self._update_with_momentum(weight_grads, bias_grads, lr)
                else:
                    self._update_sgd(weight_grads, bias_grads, lr)

                # Track batch loss
                output = self.element.forward(X_batch)
                batch_loss = self.element._binary_cross_entropy(y_batch, output)
                epoch_loss += batch_loss
                n_batches += 1

            # Record history
            if epoch % self.config.record_every == 0 or epoch == self.config.epochs - 1:
                output = self.element.forward(X)
                loss = self.element._binary_cross_entropy(y, output)
                accuracy = np.mean((output > 0.5).astype(int) == y)

                self.history['loss'].append(float(loss))
                self.history['accuracy'].append(float(accuracy))
                self.history['learning_rate'].append(float(lr))

                # Record gradient statistics (compute on full dataset for consistency)
                if self.config.record_gradients:
                    full_weight_grads, _ = self.element.backward(X, y)
                    grad_stats = {
                        'epoch': epoch,
                        'layer_grad_norms': [float(np.linalg.norm(g)) for g in full_weight_grads],
                        'max_grad': float(max(np.max(np.abs(g)) for g in full_weight_grads)),
                        'mean_grad': float(np.mean([np.mean(np.abs(g)) for g in full_weight_grads])),
                    }
                    self.history['gradient_stats'].append(grad_stats)

                # Record weight distribution statistics
                if self.config.record_weight_stats:
                    weight_stats = {
                        'epoch': epoch,
                        'layer_weight_norms': [float(np.linalg.norm(W)) for W in self.element.weights],
                        'mean_weight': float(np.mean([np.mean(W) for W in self.element.weights])),
                        'std_weight': float(np.mean([np.std(W) for W in self.element.weights])),
                    }
                    self.history['weight_stats'].append(weight_stats)

                # Validation metrics
                if X_val is not None:
                    y_val_arr = np.array(y_val).reshape(-1, 1)
                    val_output = self.element.forward(X_val)
                    val_loss = self.element._binary_cross_entropy(y_val_arr, val_output)
                    val_accuracy = np.mean((val_output > 0.5).astype(int) == y_val_arr)

                    self.history['val_loss'].append(float(val_loss))
                    self.history['val_accuracy'].append(float(val_accuracy))

                    # Early stopping check
                    if self.config.early_stopping_patience:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= self.config.early_stopping_patience:
                                if verbose:
                                    print(f"Early stopping at epoch {epoch}")
                                break

                if verbose and epoch % (self.config.record_every * 10) == 0:
                    msg = f"Epoch {epoch}: loss={loss:.4f}, acc={accuracy:.4f}"
                    if X_val is not None:
                        msg += f", val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
                    print(msg)

                # Call callbacks
                for callback in callbacks:
                    callback(epoch, self.history)

        self.element.trained = True
        self.element.history = {
            'loss': self.history['loss'],
            'accuracy': self.history['accuracy']
        }

        return self.history

    def _get_learning_rate(self, epoch: int) -> float:
        """Get learning rate for current epoch based on schedule."""
        base_lr = self.config.learning_rate

        if self.config.lr_schedule is None:
            return base_lr

        if self.config.lr_schedule == 'cosine':
            # Cosine annealing
            return base_lr * 0.5 * (1 + np.cos(np.pi * epoch / self.config.epochs))

        if self.config.lr_schedule == 'step':
            # Step decay at 50% and 75% of training
            if epoch >= 0.75 * self.config.epochs:
                return base_lr * 0.01
            if epoch >= 0.5 * self.config.epochs:
                return base_lr * 0.1
            return base_lr

        return base_lr

    def _update_sgd(
        self,
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
        lr: float
    ):
        """Standard SGD update (respects frozen layers)."""
        for i in range(len(self.element.weights)):
            # Skip frozen layers
            if hasattr(self.element, 'frozen_layers') and self.element.frozen_layers[i]:
                continue
            self.element.weights[i] -= lr * weight_grads[i]
            if self.element.biases[i] is not None:
                self.element.biases[i] -= lr * bias_grads[i]

    def _update_with_momentum(
        self,
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
        lr: float
    ):
        """SGD with momentum update (respects frozen layers)."""
        mu = self.config.momentum

        for i in range(len(self.element.weights)):
            # Skip frozen layers
            if hasattr(self.element, 'frozen_layers') and self.element.frozen_layers[i]:
                continue

            # Update velocities
            self._velocities[i] = mu * self._velocities[i] - lr * weight_grads[i]

            # Update weights
            self.element.weights[i] += self._velocities[i]

            # Biases
            if self.element.biases[i] is not None:
                self._bias_velocities[i] = mu * self._bias_velocities[i] - lr * bias_grads[i]
                self.element.biases[i] += self._bias_velocities[i]


def train_element(
    element: NeuralElement,
    X: np.ndarray,
    y: np.ndarray,
    **kwargs
) -> Dict[str, List]:
    """Convenience function to train an element."""
    config = TrainingConfig(**kwargs)
    trainer = Trainer(element, config)
    return trainer.train(X, y)


def compare_elements(
    elements: List[NeuralElement],
    X: np.ndarray,
    y: np.ndarray,
    **kwargs
) -> Dict[str, Dict]:
    """
    Train multiple elements and compare their performance.

    Returns dict mapping element names to their training histories.
    """
    results = {}

    for element in elements:
        element.reset()  # Ensure fresh start
        element.fit(X, y, **kwargs)
        results[element.name] = {
            'history': element.history.copy(),
            'final_loss': element.history['loss'][-1] if element.history['loss'] else None,
            'final_accuracy': element.history['accuracy'][-1] if element.history['accuracy'] else None,
            'params': element.config.total_params,
        }

    return results


def grid_search(
    X: np.ndarray,
    y: np.ndarray,
    depths: List[int] = [1, 2, 3],
    widths: List[int] = [2, 4, 8],
    activations: List[str] = ['relu', 'tanh', 'sigmoid'],
    n_trials: int = 3,
    **training_kwargs
) -> List[Dict]:
    """
    Grid search over element configurations.

    Trains multiple elements with different architectures and returns
    performance statistics.
    """
    results = []

    for depth in depths:
        for width in widths:
            for activation in activations:
                trial_results = []

                for trial in range(n_trials):
                    element = NeuralElement(
                        hidden_layers=[width] * depth,
                        activation=activation,
                        seed=None  # Different initialization each trial
                    )
                    element.fit(X, y, **training_kwargs)

                    trial_results.append({
                        'final_loss': element.history['loss'][-1],
                        'final_accuracy': element.history['accuracy'][-1],
                    })

                # Aggregate results
                avg_loss = np.mean([t['final_loss'] for t in trial_results])
                avg_acc = np.mean([t['final_accuracy'] for t in trial_results])
                std_loss = np.std([t['final_loss'] for t in trial_results])
                std_acc = np.std([t['final_accuracy'] for t in trial_results])

                results.append({
                    'depth': depth,
                    'width': width,
                    'activation': activation,
                    'params': (2 * width + width * width * (depth - 1) + width * 1 +
                              depth * width + 1) if depth > 0 else 3,
                    'avg_loss': float(avg_loss),
                    'std_loss': float(std_loss),
                    'avg_accuracy': float(avg_acc),
                    'std_accuracy': float(std_acc),
                    'n_trials': n_trials,
                })

    return results
