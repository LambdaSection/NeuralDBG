import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from neural.exceptions import InvalidParameterError, InvalidShapeError, ShapeMismatchError
from neural.shape_propagation.shape_propagator import ShapePropagator


def test_conv2d_kernel_too_large():
    """Error when Conv2D kernel size exceeds input dimensions"""
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


def test_dense_with_4d_input():
    """Error when Dense layer receives 4D input"""
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


def test_conv2d_missing_filters():
    """Error when Conv2D filters parameter is missing"""
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "kernel_size": (3, 3),
            "padding": "valid",
            "stride": 1
        }
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "filters" in str(excinfo.value).lower()


def test_dense_missing_units():
    """Error when Dense units parameter is missing"""
    propagator = ShapePropagator()
    input_shape = (1, 128)
    layer = {
        "type": "Dense",
        "params": {}
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "units" in str(excinfo.value).lower()


def test_conv2d_negative_filters():
    """Error when Conv2D filters is negative"""
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


def test_maxpooling2d_pool_size_too_large():
    """Error when MaxPooling2D pool_size exceeds input dimensions"""
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


def test_conv2d_negative_stride():
    """Error when Conv2D stride is negative"""
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


def test_maxpooling2d_negative_stride():
    """Error when MaxPooling2D stride is negative"""
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


def test_invalid_input_shape_empty():
    """Error when input shape is invalid (empty)"""
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


def test_invalid_input_shape_negative():
    """Error when input shape has negative dimensions"""
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


def test_missing_layer_type():
    """Error when layer type is missing"""
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
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


def test_missing_params():
    """Error when params is missing (filters parameter required)"""
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D"
    }
    with pytest.raises(InvalidParameterError) as excinfo:
        propagator.propagate(input_shape, layer, framework="tensorflow")
    assert "filters" in str(excinfo.value).lower()


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
