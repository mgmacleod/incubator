#!/usr/bin/env python3
"""
Explore Neural Elements - Basic Usage Examples

This script demonstrates how to use the Neural Elements framework
to explore and compare different neural network architectures.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.elements import NeuralElement, ElementRegistry, create_element
from src.core.activations import list_activations
from src.core.properties import compute_properties
from src.datasets.toy import get_dataset, list_datasets
from src.visualization.plots import (
    plot_decision_boundary,
    plot_training_history,
    plot_element_comparison,
    figure_to_base64
)


def example_single_element():
    """Train a single neural element and visualize its decision boundary."""
    print("\n" + "="*60)
    print("Example 1: Training a Single Neural Element")
    print("="*60)

    # Create an element with 2 hidden layers of 4 neurons each, using ReLU
    element = NeuralElement(
        hidden_layers=[4, 4],
        activation='relu',
        name='MyFirstElement'
    )

    print(f"\nCreated element: {element}")
    print(f"Total parameters: {element.config.total_params}")

    # Load the spirals dataset
    X, y = get_dataset('spirals', n_samples=200)
    print(f"Dataset: spirals ({len(y)} samples)")

    # Train the element
    print("\nTraining...")
    history = element.fit(X, y, epochs=1000, learning_rate=0.1, verbose=True)

    print(f"\nFinal loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['accuracy'][-1]*100:.1f}%")

    # Save visualization
    fig = plot_decision_boundary(element, X, y)
    fig.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
    print("\nSaved decision boundary to 'decision_boundary.png'")

    return element


def example_compare_activations():
    """Compare how different activation functions affect learning."""
    print("\n" + "="*60)
    print("Example 2: Comparing Activation Functions")
    print("="*60)

    activations = ['relu', 'tanh', 'sigmoid', 'sine', 'gelu']
    X, y = get_dataset('moons', n_samples=200)

    results = {}
    elements = []

    for act in activations:
        element = NeuralElement(
            hidden_layers=[8],
            activation=act
        )
        element.fit(X, y, epochs=500, verbose=False)

        results[act] = {
            'final_accuracy': element.history['accuracy'][-1],
            'final_loss': element.history['loss'][-1],
        }
        elements.append(element)

        print(f"{act:12} - Accuracy: {results[act]['final_accuracy']*100:5.1f}%, Loss: {results[act]['final_loss']:.4f}")

    # Create comparison plot
    fig = plot_element_comparison(elements, X, y)
    fig.savefig('activation_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison to 'activation_comparison.png'")


def example_explore_depth():
    """Explore how network depth affects capability."""
    print("\n" + "="*60)
    print("Example 3: Exploring Network Depth")
    print("="*60)

    # Use spirals - a dataset that benefits from depth
    X, y = get_dataset('spirals', n_samples=300, noise=0.15)

    print("\nTraining elements with increasing depth on spirals dataset...")
    print(f"{'Depth':<10} {'Params':<10} {'Accuracy':<10} {'Loss':<10}")
    print("-" * 40)

    elements = []

    for depth in range(1, 5):
        element = create_element(depth=depth, width=4, activation='relu')
        element.fit(X, y, epochs=1000, verbose=False)

        acc = element.history['accuracy'][-1]
        loss = element.history['loss'][-1]

        print(f"{depth:<10} {element.config.total_params:<10} {acc*100:>6.1f}%    {loss:.4f}")
        elements.append(element)

    fig = plot_element_comparison(elements, X, y)
    fig.savefig('depth_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved depth comparison to 'depth_comparison.png'")


def example_element_registry():
    """Explore the element registry (the periodic table)."""
    print("\n" + "="*60)
    print("Example 4: The Element Registry (Periodic Table)")
    print("="*60)

    registry = ElementRegistry()

    print(f"\nLoaded {len(registry)} standard elements")

    # Show elements organized by activation
    activations = ['relu', 'tanh', 'sine']
    for act in activations:
        elements = registry.list_by_activation(act)
        print(f"\n{act.upper()} elements:")
        for e in elements:
            print(f"  - {e.name}: {e.config.hidden_layers}, {e.config.total_params} params")


def example_datasets():
    """Show all available datasets."""
    print("\n" + "="*60)
    print("Example 5: Available Datasets")
    print("="*60)

    datasets = list_datasets()

    for name, info in datasets.items():
        print(f"\n{info['name']} ({name})")
        print(f"  Difficulty: {info['difficulty']}")
        print(f"  Description: {info['description']}")


def example_properties():
    """Compute and display element properties."""
    print("\n" + "="*60)
    print("Example 6: Element Properties")
    print("="*60)

    element = NeuralElement(
        hidden_layers=[8, 8],
        activation='relu'
    )

    # Train it first
    X, y = get_dataset('xor')
    element.fit(X, y, epochs=500, verbose=False)

    # Compute all properties
    props = compute_properties(element)

    print(f"\nElement: {element.name}")
    print("\nStructural Properties:")
    for key, value in props['structural'].items():
        print(f"  {key}: {value}")

    print("\nCapacity Properties:")
    for key, value in props['capacity'].items():
        print(f"  {key}: {value}")

    print("\nEmpirical Properties (after training):")
    for key, value in props['empirical'].items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  NEURAL ELEMENTS - Exploration Examples")
    print("="*60)

    # Run all examples
    example_single_element()
    example_compare_activations()
    example_explore_depth()
    example_element_registry()
    example_datasets()
    example_properties()

    print("\n" + "="*60)
    print("  Examples complete!")
    print("="*60)
    print("\nTo run the interactive web interface:")
    print("  python -m src.web.app")
    print("\nThen open http://localhost:5000 in your browser.")
