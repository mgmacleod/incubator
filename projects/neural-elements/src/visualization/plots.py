"""
Matplotlib-based visualization for neural elements.

These functions create static plots for analysis and documentation.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
import io
import base64

# Matplotlib imports with non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Custom colormaps
DECISION_CMAP = LinearSegmentedColormap.from_list(
    'decision', ['#3498db', '#ecf0f1', '#e74c3c']
)


def plot_decision_boundary(
    element,
    X: np.ndarray,
    y: np.ndarray,
    x_range: Tuple[float, float] = (-3, 3),
    y_range: Tuple[float, float] = (-3, 3),
    resolution: int = 100,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    show_probs: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the decision boundary of a neural element.

    Args:
        element: Trained NeuralElement
        X: Training data features
        y: Training data labels
        x_range: Range for x-axis
        y_range: Range for y-axis
        resolution: Grid resolution
        title: Plot title
        figsize: Figure size
        show_probs: Show probability contours vs hard boundary
        ax: Existing axes to plot on

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create mesh grid
    xx, yy = np.meshgrid(
        np.linspace(x_range[0], x_range[1], resolution),
        np.linspace(y_range[0], y_range[1], resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions
    probs = element.predict_proba(grid).reshape(xx.shape)

    # Plot decision boundary
    if show_probs:
        contour = ax.contourf(xx, yy, probs, levels=50, cmap=DECISION_CMAP, alpha=0.8)
        plt.colorbar(contour, ax=ax, label='P(class=1)')
    else:
        ax.contourf(xx, yy, (probs > 0.5).astype(int), levels=[0, 0.5, 1],
                   colors=['#3498db', '#e74c3c'], alpha=0.5)

    # Plot decision boundary line
    ax.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='white',
                        s=50, linewidths=1, zorder=10)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{element.name} Decision Boundary')

    ax.set_aspect('equal')

    return fig


def plot_training_history(
    element,
    figsize: Tuple[int, int] = (10, 4),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history (loss and accuracy).

    Args:
        element: Trained NeuralElement with history
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(len(element.history['loss']))

    # Loss plot
    ax1.plot(epochs, element.history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Accuracy plot
    ax2.plot(epochs, element.history['accuracy'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_activation_functions(
    activations: Optional[List[str]] = None,
    x_range: Tuple[float, float] = (-3, 3),
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot activation functions for comparison.

    Args:
        activations: List of activation names (None for all)
        x_range: Range for x-axis
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from ..core.activations import ACTIVATIONS

    if activations is None:
        activations = list(ACTIVATIONS.keys())

    fig, axes = plt.subplots(1, len(activations), figsize=figsize)
    if len(activations) == 1:
        axes = [axes]

    x = np.linspace(x_range[0], x_range[1], 200)

    for ax, act_name in zip(axes, activations):
        act = ACTIVATIONS[act_name]
        y = act(x)

        ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
        ax.plot(x, act.grad(x), 'r--', linewidth=1.5, alpha=0.7, label="f'(x)")

        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_title(act_name)
        ax.set_xlabel('x')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_weight_distribution(
    element,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Plot distribution of weights in each layer.

    Args:
        element: NeuralElement
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_layers = len(element.weights)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    if n_layers == 1:
        axes = [axes]

    for i, (ax, W) in enumerate(zip(axes, element.weights)):
        weights_flat = W.flatten()
        ax.hist(weights_flat, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(x=0, color='red', linewidth=1, linestyle='--')
        ax.set_title(f'Layer {i+1} ({W.shape[0]}×{W.shape[1]})')
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Count')

    plt.tight_layout()
    return fig


def plot_dataset(
    X: np.ndarray,
    y: np.ndarray,
    title: str = 'Dataset',
    figsize: Tuple[int, int] = (6, 5)
) -> plt.Figure:
    """
    Plot a 2D dataset.

    Args:
        X: Features
        y: Labels
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='white',
                        s=50, linewidths=1)
    plt.colorbar(scatter, ax=ax, label='Class')

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig


def plot_element_comparison(
    elements: List,
    X: np.ndarray,
    y: np.ndarray,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Compare decision boundaries of multiple elements.

    Args:
        elements: List of trained NeuralElements
        X: Training data features
        y: Training data labels
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_elements = len(elements)
    ncols = min(3, n_elements)
    nrows = (n_elements + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_elements == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, element in enumerate(elements):
        plot_decision_boundary(element, X, y, ax=axes[i],
                              title=f'{element.name}\nAcc: {element.history["accuracy"][-1]:.2f}')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def figure_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string for web display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_periodic_table_element(
    element,
    X: np.ndarray,
    y: np.ndarray,
    size: Tuple[int, int] = (2, 2)
) -> str:
    """
    Create a small thumbnail for periodic table display.

    Returns base64-encoded PNG.
    """
    fig, ax = plt.subplots(figsize=size)

    # Minimal decision boundary plot
    x_range = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    y_range = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    xx, yy = np.meshgrid(
        np.linspace(x_range[0], x_range[1], 50),
        np.linspace(y_range[0], y_range[1], 50)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = element.predict_proba(grid).reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=20, cmap=DECISION_CMAP, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=10, edgecolors='white', linewidths=0.5)

    ax.axis('off')
    ax.set_aspect('equal')

    return figure_to_base64(fig)
