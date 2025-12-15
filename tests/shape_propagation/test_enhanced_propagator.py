import sys
import os

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import numpy as np
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.shape_propagation.utils import calculate_memory_usage, format_memory_size
from neural.exceptions import InvalidParameterError, InvalidShapeError, ShapeMismatchError

#########################################
# 1. Test layer documentation
#########################################
def test_layer_documentation():
    propagator = ShapePropagator()

    # Get documentation for Conv2D layer
    doc = propagator.get_layer_documentation('Conv2D')

    # Check that documentation contains expected keys
    assert 'description' in doc
    assert 'parameters' in doc
    assert 'shape_transformation' in doc

    # Check that parameters include expected keys
    assert 'filters' in doc['parameters']
    assert 'kernel_size' in doc['parameters']
    assert 'padding' in doc['parameters']

    # Test formatted documentation
    formatted_doc = propagator.format_layer_documentation('Conv2D')
    assert '# Conv2D' in formatted_doc
    assert '## Parameters' in formatted_doc
    assert '## Shape Transformation' in formatted_doc

#########################################
# 2. Test shape-based error detection
#########################################
def test_shape_based_error_detection():
    propagator = ShapePropagator()

    # Create a model with a potential issue (large tensor size increase)
    input_shape = (1, 28, 28, 3)
    layers = [
        {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "padding": "same"}},
        {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same"}},
        {"type": "Flatten", "params": {}},  # This will create a large tensor
        {"type": "Dense", "params": {"units": 10}}
    ]

    # Propagate shapes
    shape = input_shape
    for layer in layers:
        shape = propagator.propagate(shape, layer, framework="tensorflow")

    # Manually create a shape history with a large tensor size change
    propagator.shape_history = [
        ("Input", (1, 28, 28, 3)),
        ("Conv2D", (1, 28, 28, 32)),
        ("Conv2D", (1, 28, 28, 64)),
        ("Flatten", (1, 50176)),  # 28*28*64 = 50176
        ("Dense", (1, 10))
    ]

    # Detect issues
    issues = propagator.detect_issues()

    # There should be at least one issue (large tensor size reduction)
    assert len(issues) > 0

    # Check that at least one issue is an info about tensor size reduction
    assert any(issue['type'] == 'info' and 'reduction' in issue['message'] for issue in issues)

#########################################
# 3. Test optimization suggestions
#########################################
def test_optimization_suggestions():
    propagator = ShapePropagator()

    # Create a model with potential optimization opportunities
    input_shape = (1, 224, 224, 3)
    layers = [
        {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same"}},
        {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same"}}
    ]

    # Propagate shapes
    shape = input_shape
    for layer in layers:
        shape = propagator.propagate(shape, layer, framework="tensorflow")

    # Get optimization suggestions
    optimizations = propagator.suggest_optimizations()

    # There should be at least one optimization suggestion (add pooling)
    assert len(optimizations) > 0

    # Check that at least one suggestion is about adding pooling
    assert any('pooling' in suggestion['message'].lower() for suggestion in optimizations)

#########################################
# 4. Test memory usage calculation
#########################################
def test_memory_usage_calculation():
    # Test with a simple shape
    shape = (1, 28, 28, 3)
    memory_bytes = calculate_memory_usage(shape)

    # Expected: 1 * 28 * 28 * 3 * 4 bytes (float32) = 9,408 bytes
    assert memory_bytes == 9408

    # Test with a larger shape
    shape = (32, 224, 224, 64)
    memory_bytes = calculate_memory_usage(shape)

    # Expected: 32 * 224 * 224 * 64 * 4 bytes = ~411 MB
    # Just check that it's in the right ballpark
    assert memory_bytes > 400000000
    assert memory_bytes < 420000000

    # Test formatting
    formatted = format_memory_size(memory_bytes)
    assert "MB" in formatted or "GB" in formatted

#########################################
# 5. Test new layer handlers
#########################################
def test_new_layer_handlers():
    propagator = ShapePropagator()

    # Test Conv1D
    input_shape = (1, 100, 32)  # (batch, steps, channels)
    layer = {"type": "Conv1D", "params": {"filters": 64, "kernel_size": 3, "padding": "same"}}
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == (1, 100, 64)

    # Test LSTM
    input_shape = (1, 100, 32)  # (batch, timesteps, features)
    layer = {"type": "LSTM", "params": {"units": 64, "return_sequences": True}}
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == (1, 100, 64)

    # Test Dropout (shape unchanged)
    input_shape = (1, 100, 64)
    layer = {"type": "Dropout", "params": {"rate": 0.5}}
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == input_shape

#########################################
# 6. Test multi-input model propagation
#########################################
def test_multi_input_model_propagation():
    propagator = ShapePropagator()

    # Define input shapes
    input_shapes = {
        "image_input": (1, 224, 224, 3),
        "metadata_input": (1, 10)
    }

    # Define model
    model_def = {
        "layers": [
            {
                "name": "conv1",
                "type": "Conv2D",
                "input": "image_input",
                "params": {"filters": 32, "kernel_size": 3, "padding": "same"}
            },
            {
                "name": "pool1",
                "type": "MaxPooling2D",
                "input": "conv1",
                "params": {"pool_size": 2}
            },
            {
                "name": "flatten",
                "type": "Flatten",
                "input": "pool1",
                "params": {}
            },
            {
                "name": "dense1",
                "type": "Dense",
                "input": "metadata_input",
                "params": {"units": 32}
            },
            {
                "name": "concat",
                "type": "Concatenate",
                "input": ["flatten", "dense1"],
                "params": {"axis": -1}
            },
            {
                "name": "output",
                "type": "Dense",
                "input": "concat",
                "params": {"units": 1}
            }
        ],
        "inputs": ["image_input", "metadata_input"],
        "outputs": ["output"]
    }

    # Propagate through the model
    output_shapes = propagator.propagate_model(input_shapes, model_def)

    # Check that output shape is correct
    assert "output" in output_shapes
    assert output_shapes["output"][1] == 1  # Output units

#########################################
# 7. Test custom layer handler registration
#########################################
def test_custom_layer_handler_registration():
    # Define a custom layer handler
    @ShapePropagator.register_layer_handler("CustomLayer")
    def handle_custom_layer(self, input_shape, params):
        units = params.get("units", 10)
        return (input_shape[0], units * 2)  # Double the units

    propagator = ShapePropagator()

    try:
        # Test the custom layer
        input_shape = (1, 100)
        layer = {"type": "CustomLayer", "params": {"units": 32}}
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")

        # Check that our custom handler was used
        assert output_shape == (1, 64)  # 32 * 2 = 64
    finally:
        # Clean up: Unregister the handler to avoid polluting other tests
        if "CustomLayer" in ShapePropagator.LAYER_HANDLERS:
            del ShapePropagator.LAYER_HANDLERS["CustomLayer"]

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
