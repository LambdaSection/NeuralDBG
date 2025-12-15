import sys
import os

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import numpy as np
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.exceptions import InvalidParameterError, InvalidShapeError, ShapeMismatchError

#########################################
# 1. Error when Conv2D kernel size exceeds input dimensions
#########################################
def test_conv2d_kernel_too_large():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": (30, 30),
            "padding": "valid",
            "stride": 1
        }
    }
    with pytest.raises(ValueError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "kernel size" in str(excinfo.value).lower()

#########################################
# 2. Error when Dense layer receives 4D input
#########################################
def test_dense_with_4d_input():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Dense",
        "params": {
            "units": 64
        }
    }
    with pytest.raises(ShapeMismatchError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "dense layer expects 2d input" in str(excinfo.value).lower()

#########################################
# 3. Error when Conv2D filters parameter is missing
#########################################
def test_conv2d_missing_filters():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            # Missing filters parameter
            "kernel_size": (3, 3),
            "padding": "valid",
            "stride": 1
        }
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "filters" in str(excinfo.value).lower()

#########################################
# 4. Error when Dense units parameter is missing
#########################################
def test_dense_missing_units():
    propagator = ShapePropagator()
    input_shape = (1, 128)
    layer = {
        "type": "Dense",
        "params": {
            # Missing units parameter
        }
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "units" in str(excinfo.value).lower()

#########################################
# 5. Error when Conv2D filters is negative
#########################################
def test_conv2d_negative_filters():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": -16,
            "kernel_size": (3, 3),
            "padding": "valid",
            "stride": 1
        }
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "positive" in str(excinfo.value).lower()

#########################################
# 6. Error when MaxPooling2D pool_size exceeds input dimensions
#########################################
def test_maxpooling2d_pool_size_too_large():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "MaxPooling2D",
        "params": {
            "pool_size": (30, 30),
            "stride": 1
        }
    }
    with pytest.raises(ShapeMismatchError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "pool_size" in str(excinfo.value).lower()

#########################################
# 7. Error when Conv2D stride is negative
#########################################
def test_conv2d_negative_stride():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": (3, 3),
            "padding": "valid",
            "stride": -1
        }
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "stride" in str(excinfo.value).lower()

#########################################
# 8. Error when MaxPooling2D stride is negative
#########################################
def test_maxpooling2d_negative_stride():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "MaxPooling2D",
        "params": {
            "pool_size": (2, 2),
            "stride": -2
        }
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "stride" in str(excinfo.value).lower()

#########################################
# 9. Error when input shape is invalid (empty)
#########################################
def test_invalid_input_shape_empty():
    propagator = ShapePropagator()
    input_shape = ()
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": (3, 3),
            "padding": "valid",
            "stride": 1
        }
    }
    with pytest.raises(InvalidShapeError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "shape" in str(excinfo.value).lower()

#########################################
# 10. Error when input shape has negative dimensions
#########################################
def test_invalid_input_shape_negative():
    propagator = ShapePropagator()
    input_shape = (1, -28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": (3, 3),
            "padding": "valid",
            "stride": 1
        }
    }
    with pytest.raises(InvalidShapeError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "negative" in str(excinfo.value).lower()

#########################################
# 11. Error when layer type is missing
#########################################
def test_missing_layer_type():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        # Missing type
        "params": {
            "filters": 16,
            "kernel_size": (3, 3),
            "padding": "valid",
            "stride": 1
        }
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "type" in str(excinfo.value).lower()

#########################################
# 12. Error when params is missing (filters parameter required)
#########################################
def test_missing_params():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D"
        # Missing params
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "filters" in str(excinfo.value).lower()

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
