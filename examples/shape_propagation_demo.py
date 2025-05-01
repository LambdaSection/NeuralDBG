#!/usr/bin/env python3
"""
Shape Propagation Demo

This script demonstrates the enhanced shape propagation capabilities of Neural,
including layer documentation, shape-based error detection, and optimization suggestions.
"""

import sys
import os
import json

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Direct import from the shape_propagator module
from neural.shape_propagation.shape_propagator import ShapePropagator

def main():
    """Run the shape propagation demo."""
    print("Neural Shape Propagation Demo")
    print("=============================\n")

    # Create a shape propagator
    propagator = ShapePropagator(debug=False)

    # Define a simple CNN model
    input_shape = (1, 224, 224, 3)  # (batch, height, width, channels)
    model = [
        {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": 2, "stride": 2}},
        {"type": "Conv2D", "params": {"filters": 64, "kernel_size": 3, "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": 2, "stride": 2}},
        {"type": "Conv2D", "params": {"filters": 128, "kernel_size": 3, "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": 2, "stride": 2}},
        {"type": "Flatten", "params": {}},
        {"type": "Dense", "params": {"units": 512}},
        {"type": "Dense", "params": {"units": 10}}
    ]

    # Propagate shapes through the model
    print("Propagating shapes through the model...")
    shape = input_shape
    for i, layer in enumerate(model):
        print(f"\nLayer {i+1}: {layer['type']}")
        print(f"  Input shape: {shape}")

        # Get layer documentation
        doc = propagator.get_layer_documentation(layer['type'])
        print(f"  Description: {doc['description']}")

        # Propagate shape
        shape = propagator.propagate(shape, layer, framework="tensorflow")
        print(f"  Output shape: {shape}")

        # Calculate memory usage
        memory_bytes = 4  # Start with 4 bytes (float32)
        for dim in shape:
            # Handle None values (use 1 as a placeholder)
            memory_bytes *= 1 if dim is None else dim
        print(f"  Memory usage: {memory_bytes / (1024 * 1024):.2f} MB")

    # Generate a report
    print("\nGenerating shape propagation report...")
    report = propagator.generate_report()

    # Display issues
    if report['issues']:
        print("\nPotential Issues Detected:")
        for issue in report['issues']:
            print(f"  [{issue['type'].upper()}] {issue['message']}")
    else:
        print("\nNo issues detected.")

    # Display optimization suggestions
    if report['optimizations']:
        print("\nOptimization Suggestions:")
        for opt in report['optimizations']:
            print(f"  [{opt['type'].upper()}] {opt['message']}")
    else:
        print("\nNo optimization suggestions.")

    # Export visualization
    print("\nExporting visualization as Mermaid diagram...")
    mermaid = propagator.export_visualization(format='mermaid')
    print(mermaid)

    print("\nDemo complete!")

if __name__ == "__main__":
    main()
