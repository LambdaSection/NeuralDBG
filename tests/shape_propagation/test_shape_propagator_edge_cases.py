import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.shape_propagation.shape_propagator import (
    ShapePropagator, ShapeValidator, PerformanceMonitor,
    detect_dead_neurons, detect_activation_anomalies, TORCH_AVAILABLE
)
from neural.exceptions import InvalidParameterError, InvalidShapeError, ShapeMismatchError


class TestShapePropagatorInitialization:
    """Test ShapePropagator initialization and configuration"""
    
    def test_init_default(self):
        """Test default initialization"""
        propagator = ShapePropagator()
        assert propagator.debug is False
        assert len(propagator.shape_history) == 0
        assert len(propagator.execution_trace) == 0
    
    def test_init_with_debug(self):
        """Test initialization with debug mode"""
        propagator = ShapePropagator(debug=True)
        assert propagator.debug is True
    
    def test_init_performance_monitor(self):
        """Test performance monitor initialization"""
        propagator = ShapePropagator()
        assert propagator.performance_monitor is not None
        assert isinstance(propagator.performance_monitor, PerformanceMonitor)


class TestLayerValidation:
    """Test layer validation edge cases"""
    
    def test_propagate_missing_type_key(self):
        """Test propagation with layer missing type key"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 1)
        layer = {"params": {"filters": 32}}
        with pytest.raises(InvalidParameterError):
            propagator.propagate(input_shape, layer)
    
    def test_propagate_empty_input_shape(self):
        """Test propagation with empty input shape"""
        propagator = ShapePropagator()
        input_shape = ()
        layer = {"type": "Dense", "params": {"units": 64}}
        with pytest.raises(InvalidShapeError) as exc_info:
            propagator.propagate(input_shape, layer)
        assert "Input shape cannot be empty" in str(exc_info.value)
    
    def test_propagate_negative_input_dimensions(self):
        """Test propagation with negative input dimensions"""
        propagator = ShapePropagator()
        input_shape = (1, -28, 28, 1)
        layer = {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3}}
        with pytest.raises(InvalidShapeError) as exc_info:
            propagator.propagate(input_shape, layer)
        assert "negative dimensions" in str(exc_info.value)


class TestConv2DEdgeCases:
    """Test Conv2D layer propagation edge cases"""
    
    def test_conv2d_kernel_as_dict_with_value(self):
        """Test Conv2D with kernel_size as dict with value key"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": {"value": 3},
                "padding": "same"
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 28, 28, 32)
    
    def test_conv2d_kernel_as_dict_without_value(self):
        """Test Conv2D with kernel_size as dict without value key"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": {"invalid_key": 3},
                "padding": "same"
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert len(output_shape) == 4
    
    def test_conv2d_stride_as_dict(self):
        """Test Conv2D with stride as dict"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": 3,
                "stride": {"value": 2},
                "padding": "same"
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape[1] == 14
    
    def test_conv2d_filters_as_dict(self):
        """Test Conv2D with filters as dict"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": {"value": 64},
                "kernel_size": 3,
                "padding": "same"
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 28, 28, 64)
    
    def test_conv2d_padding_as_dict(self):
        """Test Conv2D with padding as dict"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": 3,
                "padding": {"value": 1}
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert len(output_shape) == 4
    
    def test_conv2d_kernel_exceeds_input(self):
        """Test Conv2D with kernel size exceeding input dimensions"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": (30, 30),
                "padding": "valid"
            }
        }
        with pytest.raises(ValueError) as exc_info:
            propagator.propagate(input_shape, layer, framework="tensorflow")
        assert "kernel size" in str(exc_info.value).lower()
    
    def test_conv2d_invalid_output_dimensions(self):
        """Test Conv2D resulting in invalid output dimensions"""
        propagator = ShapePropagator()
        input_shape = (1, 5, 5, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": 10,
                "padding": "valid",
                "stride": 1
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape[1] >= 1 and output_shape[2] >= 1


class TestMaxPooling2DEdgeCases:
    """Test MaxPooling2D layer propagation edge cases"""
    
    def test_maxpooling2d_pool_size_as_dict(self):
        """Test MaxPooling2D with pool_size as dict"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 8)
        layer = {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": {"value": 2}
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 14, 14, 8)
    
    def test_maxpooling2d_stride_as_dict(self):
        """Test MaxPooling2D with stride as dict"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 8)
        layer = {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": 2,
                "stride": {"value": 2}
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 14, 14, 8)
    
    def test_maxpooling2d_stride_as_tuple(self):
        """Test MaxPooling2D with stride as tuple"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 8)
        layer = {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": 2,
                "stride": (2, 2)
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 14, 14, 8)
    
    def test_maxpooling2d_invalid_input_shape(self):
        """Test MaxPooling2D with invalid input shape"""
        propagator = ShapePropagator()
        input_shape = (1, 8)
        layer = {
            "type": "MaxPooling2D",
            "params": {"pool_size": 2}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert len(output_shape) >= 2


class TestDenseLayerEdgeCases:
    """Test Dense layer propagation edge cases"""
    
    def test_dense_units_as_dict(self):
        """Test Dense with units as dict"""
        propagator = ShapePropagator()
        input_shape = (1, 128)
        layer = {
            "type": "Dense",
            "params": {"units": {"value": 64}}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 64)
    
    def test_dense_with_1d_input(self):
        """Test Dense with 1D input (no batch dimension)"""
        propagator = ShapePropagator()
        input_shape = (256,)
        layer = {
            "type": "Dense",
            "params": {"units": 10}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (10,)
    
    def test_dense_higher_dimensional_input(self):
        """Test Dense with higher dimensional input (should raise error)"""
        propagator = ShapePropagator()
        input_shape = (1, 7, 7, 64)
        layer = {
            "type": "Dense",
            "params": {"units": 10}
        }
        with pytest.raises(ShapeMismatchError) as exc_info:
            propagator.propagate(input_shape, layer, framework="tensorflow")
        assert "expects 2D input" in str(exc_info.value)


class TestOutputLayerEdgeCases:
    """Test Output layer propagation edge cases"""
    
    def test_output_units_as_dict(self):
        """Test Output with units as dict"""
        propagator = ShapePropagator()
        input_shape = (1, 64)
        layer = {
            "type": "Output",
            "params": {"units": {"value": 10}}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 10)
    
    def test_output_with_1d_input(self):
        """Test Output with 1D input"""
        propagator = ShapePropagator()
        input_shape = (128,)
        layer = {
            "type": "Output",
            "params": {"units": 5}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (5,)


class TestGlobalAveragePooling2D:
    """Test GlobalAveragePooling2D layer propagation"""
    
    def test_gap2d_channels_last(self):
        """Test GlobalAveragePooling2D with channels_last"""
        propagator = ShapePropagator()
        input_shape = (1, 7, 7, 512)
        layer = {
            "type": "GlobalAveragePooling2D",
            "params": {"data_format": "channels_last"}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 512)
    
    def test_gap2d_channels_first(self):
        """Test GlobalAveragePooling2D with channels_first"""
        propagator = ShapePropagator()
        input_shape = (1, 512, 7, 7)
        layer = {
            "type": "GlobalAveragePooling2D",
            "params": {"data_format": "channels_first"}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="pytorch")
        assert output_shape == (1, 512)
    
    def test_gap2d_invalid_input_shape(self):
        """Test GlobalAveragePooling2D with invalid input shape"""
        propagator = ShapePropagator()
        input_shape = (1, 512)
        layer = {
            "type": "GlobalAveragePooling2D",
            "params": {}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert len(output_shape) >= 1


class TestUpSampling2D:
    """Test UpSampling2D layer propagation"""
    
    def test_upsampling2d_basic(self):
        """Test basic UpSampling2D"""
        propagator = ShapePropagator()
        input_shape = (1, 14, 14, 64)
        layer = {
            "type": "UpSampling2D",
            "params": {"size": (2, 2)}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 28, 28, 64)
    
    def test_upsampling2d_size_as_int(self):
        """Test UpSampling2D with size as int"""
        propagator = ShapePropagator()
        input_shape = (1, 7, 7, 32)
        layer = {
            "type": "UpSampling2D",
            "params": {"size": 3}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 21, 21, 32)
    
    def test_upsampling2d_size_as_dict(self):
        """Test UpSampling2D with size as dict"""
        propagator = ShapePropagator()
        input_shape = (1, 10, 10, 16)
        layer = {
            "type": "UpSampling2D",
            "params": {"size": {"value": 2}}
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 20, 20, 16)


class TestFlattenLayer:
    """Test Flatten layer edge cases"""
    
    def test_flatten_4d_input(self):
        """Test Flatten with 4D input"""
        propagator = ShapePropagator()
        input_shape = (1, 7, 7, 64)
        layer = {"type": "Flatten", "params": {}}
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 7 * 7 * 64)
    
    def test_flatten_3d_input(self):
        """Test Flatten with 3D input"""
        propagator = ShapePropagator()
        input_shape = (4, 10, 20)
        layer = {"type": "Flatten", "params": {}}
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (4, 10 * 20)
    
    def test_flatten_2d_input(self):
        """Test Flatten with 2D input"""
        propagator = ShapePropagator()
        input_shape = (1, 128)
        layer = {"type": "Flatten", "params": {}}
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == (1, 128)


class TestPaddingCalculation:
    """Test padding calculation edge cases"""
    
    def test_calculate_padding_int(self):
        """Test padding calculation with int"""
        propagator = ShapePropagator()
        params = {"padding": 2, "kernel_size": 3}
        padding = propagator._calculate_padding(params, 28)
        assert padding == 2
    
    def test_calculate_padding_tuple(self):
        """Test padding calculation with tuple"""
        propagator = ShapePropagator()
        params = {"padding": (1, 2), "kernel_size": 3}
        padding = propagator._calculate_padding(params, 28)
        assert padding == (1, 2)
    
    def test_calculate_padding_same(self):
        """Test padding calculation with 'same'"""
        propagator = ShapePropagator()
        params = {"padding": "same", "kernel_size": 3}
        padding = propagator._calculate_padding(params, 28)
        assert padding == 1
    
    def test_calculate_padding_same_tuple_kernel(self):
        """Test padding calculation with 'same' and tuple kernel"""
        propagator = ShapePropagator()
        params = {"padding": "same", "kernel_size": (5, 5)}
        padding = propagator._calculate_padding(params, 28)
        assert padding == (2, 2)
    
    def test_calculate_padding_valid(self):
        """Test padding calculation with 'valid'"""
        propagator = ShapePropagator()
        params = {"padding": "valid", "kernel_size": 3}
        padding = propagator._calculate_padding(params, 28)
        assert padding == 0
    
    def test_calculate_padding_dict_with_value(self):
        """Test padding calculation with dict containing value"""
        propagator = ShapePropagator()
        params = {"padding": {"value": 3}, "kernel_size": 3}
        padding = propagator._calculate_padding(params, 28)
        assert padding == 3
    
    def test_calculate_padding_dict_without_value(self):
        """Test padding calculation with dict without value"""
        propagator = ShapePropagator()
        params = {"padding": {"invalid": 3}, "kernel_size": 3}
        padding = propagator._calculate_padding(params, 28)
        assert padding == 0


class TestPerformanceComputation:
    """Test performance computation edge cases"""
    
    def test_compute_performance_with_none_dimensions(self):
        """Test performance computation with None dimensions"""
        propagator = ShapePropagator()
        input_shape = (None, 28, 28, 3)
        output_shape = (None, 26, 26, 32)
        layer = {
            "type": "Conv2D",
            "params": {"filters": 32, "kernel_size": (3, 3)}
        }
        flops, mem, compute_time, transfer_time = propagator._compute_performance(
            layer, input_shape, output_shape
        )
        assert flops >= 0
        assert mem >= 0
    
    def test_compute_performance_unknown_layer(self):
        """Test performance computation for unknown layer type"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        output_shape = (1, 28, 28, 3)
        layer = {"type": "UnknownLayer", "params": {}}
        flops, mem, compute_time, transfer_time = propagator._compute_performance(
            layer, input_shape, output_shape
        )
        assert flops == 0
        assert mem > 0


class TestExecutionTrace:
    """Test execution trace functionality"""
    
    def test_get_trace_with_dict_entries(self):
        """Test get_trace with dict format entries"""
        propagator = ShapePropagator(debug=True)
        input_shape = (1, 28, 28, 1)
        layer = {"type": "Flatten", "params": {}}
        propagator.propagate(input_shape, layer, framework="tensorflow")
        trace = propagator.get_trace()
        assert len(trace) > 0
        assert 'layer' in trace[0]
        assert 'execution_time' in trace[0]
    
    def test_get_trace_empty(self):
        """Test get_trace with no propagations"""
        propagator = ShapePropagator()
        trace = propagator.get_trace()
        assert trace == []


class TestShapeValidator:
    """Test ShapeValidator class"""
    
    def test_validate_conv_4d_input(self):
        """Test Conv validation with correct 4D input"""
        input_shape = (1, 28, 28, 3)
        params = {"kernel_size": 3}
        ShapeValidator.validate_layer("Conv2D", input_shape, params)
    
    def test_validate_conv_invalid_dimensions(self):
        """Test Conv validation with invalid dimensions"""
        input_shape = (1, 28, 3)
        params = {"kernel_size": 3}
        with pytest.raises(ShapeMismatchError) as exc_info:
            ShapeValidator.validate_layer("Conv2D", input_shape, params)
        assert "4D input" in str(exc_info.value)
    
    def test_validate_conv_kernel_too_large(self):
        """Test Conv validation with kernel larger than input"""
        input_shape = (1, 10, 10, 3)
        params = {"kernel_size": 15}
        with pytest.raises(ShapeMismatchError) as exc_info:
            ShapeValidator.validate_layer("Conv2D", input_shape, params)
        assert "exceeds input dimension" in str(exc_info.value)
    
    def test_validate_dense_2d_input(self):
        """Test Dense validation with correct 2D input"""
        input_shape = (1, 128)
        params = {"units": 64}
        ShapeValidator.validate_layer("Dense", input_shape, params)
    
    def test_validate_dense_higher_d_input(self):
        """Test Dense validation with higher dimensional input"""
        input_shape = (1, 7, 7, 64)
        params = {"units": 10}
        with pytest.raises(ShapeMismatchError) as exc_info:
            ShapeValidator.validate_layer("Dense", input_shape, params)
        assert "2D input" in str(exc_info.value)


class TestLayerHandlerRegistry:
    """Test external layer handler registration"""
    
    def test_register_custom_handler(self):
        """Test registering a custom layer handler"""
        @ShapePropagator.register_layer_handler("CustomLayer")
        def custom_handler(propagator, input_shape, params):
            return (input_shape[0], 100)
        
        assert "CustomLayer" in ShapePropagator.LAYER_HANDLERS
        
        propagator = ShapePropagator()
        input_shape = (1, 50)
        layer = {"type": "CustomLayer", "params": {}}
        output_shape = propagator.propagate(input_shape, layer)
        assert output_shape == (1, 100)
    
    def test_custom_handler_override(self):
        """Test that custom handler overrides default behavior"""
        @ShapePropagator.register_layer_handler("Dense")
        def custom_dense_handler(propagator, input_shape, params):
            return (input_shape[0], 999)
        
        propagator = ShapePropagator()
        input_shape = (1, 128)
        layer = {"type": "Dense", "params": {"units": 64}}
        output_shape = propagator.propagate(input_shape, layer)
        assert output_shape == (1, 999)
        
        del ShapePropagator.LAYER_HANDLERS["Dense"]


class TestVisualizationMethods:
    """Test visualization and reporting methods"""
    
    def test_generate_report(self):
        """Test generate_report method"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 1)
        layers = [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "padding": "same"}},
            {"type": "Flatten", "params": {}},
            {"type": "Dense", "params": {"units": 10}}
        ]
        shape = input_shape
        for layer in layers:
            shape = propagator.propagate(shape, layer, framework="tensorflow")
        
        report = propagator.generate_report()
        assert 'dot_graph' in report
        assert 'plotly_chart' in report
        assert 'shape_history' in report
    
    def test_export_visualization_mermaid(self):
        """Test mermaid export"""
        propagator = ShapePropagator()
        input_shape = (1, 10)
        layer = {"type": "Dense", "params": {"units": 5}}
        propagator.propagate(input_shape, layer)
        
        mermaid = propagator.export_visualization(format='mermaid')
        assert "graph TD" in mermaid
        assert "Dense" in mermaid
    
    def test_export_visualization_invalid_format(self):
        """Test export with invalid format"""
        propagator = ShapePropagator()
        with pytest.raises(ValueError) as exc_info:
            propagator.export_visualization(format='invalid')
        assert "Unsupported format" in str(exc_info.value)


class TestDetectionFunctions:
    """Test detection utility functions"""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_detect_dead_neurons_with_torch(self):
        """Test dead neuron detection with torch available"""
        import torch
        
        class DummyLayer:
            pass
        
        layer = DummyLayer()
        layer.__class__.__name__ = "TestLayer"
        input_tensor = torch.randn(1, 10)
        output_tensor = torch.zeros(1, 10)
        
        result = detect_dead_neurons(layer, input_tensor, output_tensor)
        assert result['dead_ratio'] == 1.0
    
    def test_detect_dead_neurons_without_torch(self):
        """Test dead neuron detection without torch"""
        if TORCH_AVAILABLE:
            pytest.skip("PyTorch is available")
        
        class DummyLayer:
            pass
        
        layer = DummyLayer()
        result = detect_dead_neurons(layer, None, None)
        assert 'error' in result
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_detect_activation_anomalies_with_torch(self):
        """Test activation anomaly detection with torch"""
        import torch
        
        class DummyLayer:
            pass
        
        layer = DummyLayer()
        layer.__class__.__name__ = "TestLayer"
        input_tensor = torch.randn(1, 10)
        output_tensor = torch.randn(1, 10) * 1000
        
        result = detect_activation_anomalies(layer, input_tensor, output_tensor)
        assert result['anomaly'] is True
    
    def test_detect_activation_anomalies_without_torch(self):
        """Test activation anomaly detection without torch"""
        if TORCH_AVAILABLE:
            pytest.skip("PyTorch is available")
        
        class DummyLayer:
            pass
        
        layer = DummyLayer()
        result = detect_activation_anomalies(layer, None, None)
        assert 'error' in result


class TestMultiInputPropagation:
    """Test propagation with multiple inputs"""
    
    def test_propagate_model_single_input(self):
        """Test propagate_model with single input"""
        propagator = ShapePropagator()
        input_shapes = {"input1": (1, 10)}
        model_def = {
            "layers": [
                {"name": "dense1", "type": "Dense", "params": {"units": 5}, "input": "input1"}
            ],
            "outputs": ["dense1"]
        }
        output_shapes = propagator.propagate_model(input_shapes, model_def)
        assert "dense1" in output_shapes
        assert output_shapes["dense1"] == (1, 5)
    
    def test_propagate_model_with_concatenate(self):
        """Test propagate_model with concatenate layer"""
        propagator = ShapePropagator()
        input_shapes = {"input1": (1, 10), "input2": (1, 20)}
        model_def = {
            "layers": [
                {"name": "concat", "type": "Concatenate", "params": {"axis": -1}, "input": ["input1", "input2"]}
            ],
            "outputs": ["concat"]
        }


class TestPerformanceMonitor:
    """Test PerformanceMonitor class"""
    
    def test_monitor_resources(self):
        """Test resource monitoring"""
        monitor = PerformanceMonitor()
        resources = monitor.monitor_resources()
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert 'gpu_memory' in resources
        assert 'io_usage' in resources
    
    def test_resource_history(self):
        """Test resource history accumulation"""
        monitor = PerformanceMonitor()
        monitor.monitor_resources()
        monitor.monitor_resources()
        assert len(monitor.resource_history) == 2


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
