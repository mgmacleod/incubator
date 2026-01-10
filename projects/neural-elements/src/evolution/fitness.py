"""
Fitness evaluation for evolutionary neural architecture search.

Implements multi-objective fitness scoring with a weighted sum for selection.

Fitness components:
- test_accuracy: Primary objective (generalization performance)
- parameter_efficiency: Accuracy relative to model complexity
- convergence_speed: How quickly the network learns
- noise_robustness: Stability under input perturbation
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .genome import Genome


@dataclass
class FitnessScores:
    """
    Multi-objective fitness scores for a genome.

    All scores are normalized to [0, 1] range where higher is better.
    """
    test_accuracy: float          # Accuracy on held-out test set
    parameter_efficiency: float   # accuracy / log10(params)
    convergence_speed: float      # 1 / epochs_to_90_percent
    noise_robustness: float       # Accuracy with noisy inputs
    training_failed: bool         # Whether training diverged or failed

    # Optional detailed metrics
    train_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    epochs_to_convergence: Optional[int] = None
    total_params: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FitnessScores':
        """Create from dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        if self.training_failed:
            return "FitnessScores(FAILED)"
        return (
            f"FitnessScores(acc={self.test_accuracy:.3f}, "
            f"eff={self.parameter_efficiency:.3f}, "
            f"speed={self.convergence_speed:.3f}, "
            f"robust={self.noise_robustness:.3f})"
        )


# Fitness weights for weighted sum (must sum to 1.0)
FITNESS_WEIGHTS = {
    'test_accuracy': 0.60,
    'parameter_efficiency': 0.20,
    'convergence_speed': 0.10,
    'noise_robustness': 0.10,
}


def weighted_fitness(scores: FitnessScores) -> float:
    """
    Compute single scalar fitness from multi-objective scores.

    Uses weighted sum:
    - 60% test accuracy (primary goal)
    - 20% parameter efficiency (practical deployment)
    - 10% convergence speed (training cost)
    - 10% noise robustness (generalization)

    Args:
        scores: FitnessScores object

    Returns:
        Scalar fitness value in [0, 1] range
    """
    if scores.training_failed:
        return 0.0

    return (
        FITNESS_WEIGHTS['test_accuracy'] * scores.test_accuracy +
        FITNESS_WEIGHTS['parameter_efficiency'] * min(scores.parameter_efficiency, 1.0) +
        FITNESS_WEIGHTS['convergence_speed'] * min(scores.convergence_speed, 1.0) +
        FITNESS_WEIGHTS['noise_robustness'] * scores.noise_robustness
    )


def compute_fitness(
    genome: 'Genome',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 500,
    learning_rate: Optional[float] = None,
    record_every: int = 10,
) -> FitnessScores:
    """
    Evaluate a genome's fitness by training and testing.

    Args:
        genome: Architecture to evaluate
        X_train: Training features (N x 2)
        y_train: Training labels (N x 1)
        X_test: Test features (M x 2)
        y_test: Test labels (M x 1)
        epochs: Number of training epochs
        learning_rate: Override genome's learning rate if provided
        record_every: How often to record training history

    Returns:
        FitnessScores with all objectives evaluated
    """
    # Import here to avoid circular dependencies
    from ..core.elements import NeuralElement
    from ..core.training import Trainer, TrainingConfig
    from ..core.activations import get_activation

    lr = learning_rate if learning_rate is not None else genome.learning_rate

    try:
        # Create element from genome
        element = NeuralElement(
            hidden_layers=genome.hidden_layers,
            activation=genome.activations[0],  # Primary activation
            skip_connections=genome.skip_connections,
        )

        # If mixed activations, we need to manually set per-layer activations
        # This is a workaround since NeuralElement doesn't support mixed natively
        if genome.has_mixed_activations():
            element._layer_activations = [
                get_activation(act) for act in genome.activations
            ]

        # Train
        config = TrainingConfig(
            epochs=epochs,
            learning_rate=lr,
            record_every=record_every,
        )
        trainer = Trainer(element, config)
        history = trainer.train(X_train, y_train)

        # Get final training accuracy
        train_output = element.forward(X_train)
        train_predictions = (train_output > 0.5).astype(int).flatten()
        train_accuracy = float(np.mean(train_predictions == y_train.flatten()))

        # Evaluate on test set
        test_output = element.forward(X_test)
        test_predictions = (test_output > 0.5).astype(int).flatten()
        test_accuracy = float(np.mean(test_predictions == y_test.flatten()))

        # Parameter efficiency: accuracy / log10(params)
        # Normalized so a 100-param network at 100% accuracy scores 0.5
        param_efficiency = test_accuracy / (np.log10(genome.total_params + 1) / 2)

        # Convergence speed: find epochs to reach 90% of final accuracy
        final_acc = history['accuracy'][-1] if history['accuracy'] else 0
        target = 0.9 * final_acc
        epochs_to_target = epochs  # Default if never reached
        for i, acc in enumerate(history['accuracy']):
            if acc >= target and target > 0.5:  # Only count if meaningful
                epochs_to_target = (i + 1) * record_every
                break
        # Normalize: faster convergence = higher score
        # Max score 1.0 at 10 epochs, approaches 0 at 500 epochs
        convergence_speed = 1.0 / (1.0 + epochs_to_target / 50.0)

        # Noise robustness: test with Gaussian noise
        noise_std = 0.1
        X_noisy = X_test + np.random.normal(0, noise_std, X_test.shape)
        noisy_output = element.forward(X_noisy)
        noisy_predictions = (noisy_output > 0.5).astype(int).flatten()
        noise_robustness = float(np.mean(noisy_predictions == y_test.flatten()))

        return FitnessScores(
            test_accuracy=test_accuracy,
            parameter_efficiency=min(param_efficiency, 1.0),
            convergence_speed=convergence_speed,
            noise_robustness=noise_robustness,
            training_failed=False,
            train_accuracy=train_accuracy,
            final_loss=history['loss'][-1] if history['loss'] else None,
            epochs_to_convergence=epochs_to_target,
            total_params=genome.total_params,
        )

    except Exception as e:
        # Training failed (diverged, NaN, etc.)
        return FitnessScores(
            test_accuracy=0.0,
            parameter_efficiency=0.0,
            convergence_speed=0.0,
            noise_robustness=0.0,
            training_failed=True,
        )


def quick_fitness_screen(
    genome: 'Genome',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    quick_epochs: int = 50,
    threshold: float = 0.6,
) -> bool:
    """
    Quick screening to eliminate clearly unfit genomes.

    Trains for a small number of epochs and checks if accuracy
    exceeds a threshold. Used to avoid wasting compute on bad
    architectures.

    Args:
        genome: Architecture to screen
        X_train, y_train: Training data
        X_test, y_test: Test data
        quick_epochs: Number of epochs for quick training
        threshold: Minimum accuracy to pass screening

    Returns:
        True if genome passes screening, False otherwise
    """
    scores = compute_fitness(
        genome,
        X_train, y_train,
        X_test, y_test,
        epochs=quick_epochs,
        record_every=quick_epochs,  # Only record final
    )

    if scores.training_failed:
        return False

    return scores.test_accuracy >= threshold


def compare_fitness(scores1: FitnessScores, scores2: FitnessScores) -> int:
    """
    Compare two fitness scores.

    Returns:
        1 if scores1 is better
        -1 if scores2 is better
        0 if equal (within tolerance)
    """
    if scores1.training_failed and scores2.training_failed:
        return 0
    if scores1.training_failed:
        return -1
    if scores2.training_failed:
        return 1

    f1 = weighted_fitness(scores1)
    f2 = weighted_fitness(scores2)

    if abs(f1 - f2) < 0.001:  # Within tolerance
        return 0
    return 1 if f1 > f2 else -1
