import sys
import os

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import numpy as np
from neural.shape_propagation.shape_propagator import ShapePropagator

#########################################
# 1. Conv2D with HPO filters parameter
#########################################
def test_conv2d_hpo_filters():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": {"value": 32},  # HPO parameter as dict with 'value' key
            "kernel_size": (3, 3),
            "padding": "same",
            "stride": 1
        }
    }
    expected = (1, 28, 28, 32)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 2. Conv2D with HPO filters parameter without 'value' key
#########################################
def test_conv2d_hpo_filters_no_value_key():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": {"min": 16, "max": 64},  # HPO parameter as dict without 'value' key
            "kernel_size": (3, 3),
            "padding": "same",
            "stride": 1
        }
    }
    # Should use default value (32)
    expected = (1, 28, 28, 32)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 3. MaxPooling2D with HPO pool_size parameter
#########################################
def test_maxpooling2d_hpo_pool_size():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 16)
    layer = {
        "type": "MaxPooling2D",
        "params": {
            "pool_size": {"value": 2},  # HPO parameter as dict with 'value' key
            "stride": 2,
            "data_format": "channels_last"
        }
    }
    expected = (1, 14, 14, 16)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 4. MaxPooling2D with HPO pool_size parameter without 'value' key
#########################################
def test_maxpooling2d_hpo_pool_size_no_value_key():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 16)
    layer = {
        "type": "MaxPooling2D",
        "params": {
            "pool_size": {"min": 2, "max": 4},  # HPO parameter as dict without 'value' key
            "stride": 2,
            "data_format": "channels_last"
        }
    }
    # Should use default value (2)
    expected = (1, 14, 14, 16)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 5. Dense with HPO units parameter
#########################################
def test_dense_hpo_units():
    propagator = ShapePropagator()
    input_shape = (1, 128)
    layer = {
        "type": "Dense",
        "params": {
            "units": {"value": 64}  # HPO parameter as dict with 'value' key
        }
    }
    expected = (1, 64)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 6. Dense with HPO units parameter without 'value' key
#########################################
def test_dense_hpo_units_no_value_key():
    propagator = ShapePropagator()
    input_shape = (1, 128)
    layer = {
        "type": "Dense",
        "params": {
            "units": {"min": 32, "max": 128}  # HPO parameter as dict without 'value' key
        }
    }
    # Should use default value (64)
    expected = (1, 64)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 7. Output layer with HPO units parameter
#########################################
def test_output_hpo_units():
    propagator = ShapePropagator()
    input_shape = (1, 64)
    layer = {
        "type": "Output",
        "params": {
            "units": {"value": 10}  # HPO parameter as dict with 'value' key
        }
    }
    expected = (1, 10)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 8. Output layer with HPO units parameter without 'value' key
#########################################
def test_output_hpo_units_no_value_key():
    propagator = ShapePropagator()
    input_shape = (1, 64)
    layer = {
        "type": "Output",
        "params": {
            "units": {"min": 5, "max": 20}  # HPO parameter as dict without 'value' key
        }
    }
    # Should use default value (10)
    expected = (1, 10)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 9. Conv2D with multiple HPO parameters
#########################################
def test_conv2d_multiple_hpo_params():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": {"value": 32},
            "kernel_size": {"value": 3},  # HPO parameter for kernel_size
            "padding": "same",
            "stride": {"value": 1}  # HPO parameter for stride
        }
    }
    expected = (1, 28, 28, 32)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 10. Complex model with multiple HPO parameters
#########################################
def test_complex_model_with_hpo():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layers = [
        {
            "type": "Conv2D",
            "params": {
                "filters": {"value": 32},
                "kernel_size": (3, 3),
                "padding": "same",
                "stride": 1
            }
        },
        {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": {"value": 2},
                "stride": 2
            }
        },
        {
            "type": "Conv2D",
            "params": {
                "filters": {"value": 64},
                "kernel_size": (3, 3),
                "padding": "same",
                "stride": 1
            }
        },
        {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": {"value": 2},
                "stride": 2
            }
        },
        {
            "type": "Flatten",
            "params": {}
        },
        {
            "type": "Dense",
            "params": {
                "units": {"value": 128}
            }
        },
        {
            "type": "Output",
            "params": {
                "units": {"value": 10}
            }
        }
    ]

    shape = input_shape
    for layer in layers:
        shape = propagator.propagate(shape, layer, framework="tensorflow")

    # Expected shape after all layers
    # Conv2D: (1, 28, 28, 32)
    # MaxPooling2D: (1, 14, 14, 32)
    # Conv2D: (1, 14, 14, 64)
    # MaxPooling2D: (1, 7, 7, 64)
    # Flatten: (1, 7*7*64) = (1, 3136)
    # Dense: (1, 128)
    # Output: (1, 10)
    expected = (1, 10)
    assert shape == expected

#########################################
# 11. Conv2D with HPO kernel_size as integer
#########################################
def test_conv2d_hpo_kernel_size_int():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 32,
            "kernel_size": {"value": 3},  # HPO parameter with integer value
            "padding": "valid",
            "stride": 1
        }
    }
    # Should handle the HPO parameter with integer value
    expected = (1, 26, 26, 32)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 12. MaxPooling2D with HPO stride parameter
#########################################
def test_maxpooling2d_hpo_stride():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 16)
    layer = {
        "type": "MaxPooling2D",
        "params": {
            "pool_size": (2, 2),
            "stride": {"value": 2},  # HPO parameter for stride
            "data_format": "channels_last"
        }
    }
    expected = (1, 14, 14, 16)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
