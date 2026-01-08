"""
Flask web application for Neural Elements.

Provides an interactive interface to explore the periodic table of neural networks.
"""

import os
import json
from flask import Flask, render_template, jsonify, request
import numpy as np

from ..core.elements import NeuralElement, ElementRegistry, create_element
from ..core.activations import ACTIVATIONS, list_activations
from ..core.properties import compute_properties
from ..datasets.toy import DATASETS, get_dataset, list_datasets
from ..visualization.interactive import (
    generate_decision_boundary_data,
    generate_training_animation_data,
    generate_periodic_table_data,
    generate_element_card_data,
)


def create_app():
    """Create and configure the Flask application."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    template_dir = os.path.join(project_root, 'web', 'templates')
    static_dir = os.path.join(project_root, 'web', 'static')

    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir)

    # Initialize element registry
    registry = ElementRegistry()

    # Current state
    state = {
        'current_element': None,
        'current_dataset': 'spirals',
        'X': None,
        'y': None,
    }

    # Load default dataset
    state['X'], state['y'] = get_dataset('spirals')

    @app.route('/')
    def index():
        """Main page with periodic table view."""
        return render_template('index.html')

    @app.route('/api/periodic-table')
    def api_periodic_table():
        """Get periodic table data."""
        table_data = generate_periodic_table_data(registry)
        return jsonify(table_data)

    @app.route('/api/elements')
    def api_elements():
        """List all elements."""
        elements = [e.to_dict() for e in registry]
        return jsonify({'elements': elements})

    @app.route('/api/element/<name>')
    def api_element(name):
        """Get details for a specific element."""
        element = registry.get(name)
        if element is None:
            return jsonify({'error': f'Element {name} not found'}), 404
        return jsonify(generate_element_card_data(element))

    @app.route('/api/activations')
    def api_activations():
        """List available activation functions."""
        return jsonify(list_activations())

    @app.route('/api/datasets')
    def api_datasets():
        """List available datasets."""
        return jsonify(list_datasets())

    @app.route('/api/dataset/<name>')
    def api_dataset(name):
        """Get a dataset."""
        try:
            X, y = get_dataset(name)
            state['X'], state['y'] = X, y
            state['current_dataset'] = name
            return jsonify({
                'name': name,
                'n_samples': len(y),
                'x': X[:, 0].tolist(),
                'y': X[:, 1].tolist(),
                'labels': y.tolist(),
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 404

    @app.route('/api/create-element', methods=['POST'])
    def api_create_element():
        """Create a new element."""
        data = request.json
        try:
            element = NeuralElement(
                hidden_layers=data.get('hidden_layers', [4]),
                activation=data.get('activation', 'relu'),
                input_dim=data.get('input_dim', 2),
                output_dim=data.get('output_dim', 1),
            )
            registry.register(element)
            state['current_element'] = element
            return jsonify(generate_element_card_data(element))
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/api/train', methods=['POST'])
    def api_train():
        """Train an element and return training animation data."""
        data = request.json

        # Get or create element
        element_name = data.get('element')
        if element_name:
            element = registry.get(element_name)
            if element is None:
                return jsonify({'error': f'Element {element_name} not found'}), 404
            element = element.copy()  # Create fresh copy for training
        else:
            # Create element from config
            hidden_layers = data.get('hidden_layers', [4])
            activation = data.get('activation', 'relu')
            element = NeuralElement(hidden_layers=hidden_layers, activation=activation)

        # Get dataset
        dataset_name = data.get('dataset', state['current_dataset'])
        try:
            X, y = get_dataset(dataset_name)
        except ValueError:
            X, y = state['X'], state['y']

        # Training parameters
        epochs = min(data.get('epochs', 500), 2000)  # Cap at 2000
        record_every = data.get('record_every', 10)

        # Generate training animation data
        animation_data = generate_training_animation_data(
            element, X, y,
            epochs=epochs,
            record_every=record_every
        )

        # Add element info
        animation_data['element'] = generate_element_card_data(element)

        return jsonify(animation_data)

    @app.route('/api/decision-boundary', methods=['POST'])
    def api_decision_boundary():
        """Get decision boundary for a trained element."""
        data = request.json

        element_name = data.get('element')
        if element_name:
            element = registry.get(element_name)
            if element is None:
                return jsonify({'error': f'Element {element_name} not found'}), 404
        elif state['current_element']:
            element = state['current_element']
        else:
            return jsonify({'error': 'No element specified'}), 400

        boundary_data = generate_decision_boundary_data(element)
        return jsonify(boundary_data)

    @app.route('/api/compare', methods=['POST'])
    def api_compare():
        """Compare multiple elements on a dataset."""
        data = request.json

        element_configs = data.get('elements', [
            {'hidden_layers': [4], 'activation': 'relu'},
            {'hidden_layers': [4, 4], 'activation': 'relu'},
            {'hidden_layers': [4], 'activation': 'tanh'},
        ])

        dataset_name = data.get('dataset', 'spirals')
        epochs = min(data.get('epochs', 500), 1000)

        X, y = get_dataset(dataset_name)

        results = []
        for config in element_configs:
            element = NeuralElement(**config)
            element.fit(X, y, epochs=epochs, verbose=False)

            boundary_data = generate_decision_boundary_data(element)
            results.append({
                'element': generate_element_card_data(element),
                'boundary': boundary_data,
            })

        return jsonify({
            'results': results,
            'dataset': {
                'name': dataset_name,
                'x': X[:, 0].tolist(),
                'y': X[:, 1].tolist(),
                'labels': y.tolist(),
            }
        })

    @app.route('/api/properties/<name>')
    def api_properties(name):
        """Get properties for an element."""
        element = registry.get(name)
        if element is None:
            return jsonify({'error': f'Element {name} not found'}), 404
        return jsonify(compute_properties(element))

    return app


def main():
    """Run the Flask development server."""
    app = create_app()
    print("\n" + "="*60)
    print("Neural Elements - Periodic Table of Neural Networks")
    print("="*60)
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000)


if __name__ == '__main__':
    main()
