#!/usr/bin/env python3
"""
Quick Start - Minimal example to get started with Neural Elements.

Run this script to see a neural element learn in action.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.elements import NeuralElement
from src.datasets.toy import get_dataset

print("Neural Elements - Quick Start")
print("="*40)

# Create a simple neural element
element = NeuralElement(
    hidden_layers=[4, 4],  # 2 hidden layers with 4 neurons each
    activation='relu'       # ReLU activation
)
print(f"\nCreated: {element}")

# Load a toy dataset
X, y = get_dataset('xor')
print(f"Dataset: XOR ({len(y)} samples)")

# Train
print("\nTraining...")
element.fit(X, y, epochs=500, verbose=True)

print(f"\nFinal accuracy: {element.history['accuracy'][-1]*100:.1f}%")
print("\nSuccess! The element learned the XOR function.")
print("\nTry modifying the hidden_layers or activation to experiment!")
