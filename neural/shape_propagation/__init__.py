"""
Neural Shape Propagation Module.

This module provides tools for propagating tensor shapes through neural network models,
detecting potential issues, and suggesting optimizations.

The shape propagation system:
    - Validates tensor dimensions through each layer
    - Detects shape mismatches and incompatibilities
    - Suggests architecture optimizations
    - Monitors memory usage and computational requirements
    - Supports TensorFlow, PyTorch, and JAX frameworks

Classes
-------
ShapePropagator
    Main class for tensor shape inference and validation

Examples
--------
>>> from neural.shape_propagation import ShapePropagator
>>> propagator = ShapePropagator(debug=False)
>>> input_shape = (None, 3, 224, 224)
>>> layer = {"type": "Conv2D", "params": {"filters": 64, "kernel_size": 3}}
>>> output_shape = propagator.propagate(input_shape, layer)
"""

from .shape_propagator import ShapePropagator

__all__ = ['ShapePropagator']
