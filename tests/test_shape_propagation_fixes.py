"""
Test suite for shape propagation bug fixes.

This test file verifies fixes for:
1. Conv2D with various padding configurations (same, valid, explicit padding)
2. LSTM/GRU return_sequences shape handling
3. Transformer layer shape propagation
4. Multi-input and concatenation layer shapes
5. Proper handling of None dimensions
6. FLOPs and memory calculations
"""

import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.shape_propagation.layer_handlers import (
    handle_concatenate, handle_add, handle_lstm, handle_gru
)


class TestConv2DPaddingConfigurations:
    """Test Conv2D with various padding configurations"""
    
    def test_conv2d_valid_padding(self):
        """Test Conv2D with 'valid' padding"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 16,
                "kernel_size": (3, 3),
                "padding": "valid",
                "stride": 1
            }
        }
        # (28 - 3) // 1 + 1 = 26
        expected = (1, 26, 26, 16)
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_conv2d_same_padding_stride1(self):
        """Test Conv2D with 'same' padding and stride=1"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 16,
                "kernel_size": (3, 3),
                "padding": "same",
                "stride": 1
            }
        }
        # Same padding with stride=1 maintains spatial dimensions
        expected = (1, 28, 28, 16)
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_conv2d_same_padding_stride2(self):
        """Test Conv2D with 'same' padding and stride=2"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 16,
                "kernel_size": (3, 3),
                "padding": "same",
                "stride": 2
            }
        }
        # Same padding with stride=2: output = ceil(28/2) = 14
        expected = (1, 14, 14, 16)
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_conv2d_explicit_padding(self):
        """Test Conv2D with explicit integer padding"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 16,
                "kernel_size": (3, 3),
                "padding": 2,  # explicit padding
                "stride": 1
            }
        }
        # (28 + 2*2 - 3) // 1 + 1 = 30
        expected = (1, 30, 30, 16)
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_conv2d_channels_first_same_padding(self):
        """Test Conv2D with channels_first and same padding"""
        propagator = ShapePropagator()
        input_shape = (1, 3, 28, 28)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 16,
                "kernel_size": (5, 5),
                "padding": "same",
                "stride": 1,
                "data_format": "channels_first"
            }
        }
        expected = (1, 16, 28, 28)
        output_shape = propagator.propagate(input_shape, layer, framework="pytorch")
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_conv2d_asymmetric_kernel_same_padding(self):
        """Test Conv2D with asymmetric kernel and same padding"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": (3, 5),
                "padding": "same",
                "stride": 1
            }
        }
        expected = (1, 28, 28, 32)
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"


class TestLSTMGRUReturnSequences:
    """Test LSTM/GRU return_sequences shape handling"""
    
    def test_lstm_return_sequences_true(self):
        """Test LSTM with return_sequences=True"""
        input_shape = (None, 10, 64)  # (batch, seq_len, input_size)
        params = {
            "units": 128,
            "return_sequences": True
        }
        output_shape = handle_lstm(input_shape, params)
        expected = (None, 10, 128)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_lstm_return_sequences_false(self):
        """Test LSTM with return_sequences=False"""
        input_shape = (None, 10, 64)
        params = {
            "units": 128,
            "return_sequences": False
        }
        output_shape = handle_lstm(input_shape, params)
        expected = (None, 128)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_gru_return_sequences_true(self):
        """Test GRU with return_sequences=True"""
        input_shape = (None, 20, 32)
        params = {
            "units": 64,
            "return_sequences": True
        }
        output_shape = handle_gru(input_shape, params)
        expected = (None, 20, 64)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_gru_return_sequences_false(self):
        """Test GRU with return_sequences=False"""
        input_shape = (None, 20, 32)
        params = {
            "units": 64,
            "return_sequences": False
        }
        output_shape = handle_gru(input_shape, params)
        expected = (None, 64)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_lstm_with_batch_size(self):
        """Test LSTM with explicit batch size"""
        input_shape = (32, 15, 128)
        params = {
            "units": 256,
            "return_sequences": True
        }
        output_shape = handle_lstm(input_shape, params)
        expected = (32, 15, 256)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"


class TestTransformerLayers:
    """Test transformer layer shape propagation"""
    
    def test_transformer_encoder_tensorflow(self):
        """Test TransformerEncoder with TensorFlow framework"""
        propagator = ShapePropagator()
        input_shape = (None, 50, 512)  # (batch, seq_len, d_model)
        layer = {
            "type": "TransformerEncoder",
            "params": {
                "num_heads": 8,
                "d_model": 512
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        # TensorFlow preserves full shape
        expected = (None, 50, 512)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_transformer_encoder_pytorch(self):
        """Test TransformerEncoder with PyTorch framework"""
        propagator = ShapePropagator()
        input_shape = (None, 50, 512)
        layer = {
            "type": "TransformerEncoder",
            "params": {
                "num_heads": 8,
                "d_model": 512
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="pytorch")
        # PyTorch returns (batch, seq_len)
        expected = (None, 50)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_multihead_attention(self):
        """Test MultiHeadAttention shape preservation"""
        propagator = ShapePropagator()
        input_shape = (None, 100, 256)
        layer = {
            "type": "MultiHeadAttention",
            "params": {
                "num_heads": 4,
                "key_dim": 64
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        # MultiHeadAttention preserves shape
        expected = (None, 100, 256)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"


class TestMultiInputConcatenation:
    """Test multi-input and concatenation layer shapes"""
    
    def test_concatenate_axis_last(self):
        """Test Concatenate along last axis (default)"""
        input_shapes = [
            (None, 10, 20),
            (None, 10, 30),
            (None, 10, 50)
        ]
        params = {"axis": -1}
        output_shape = handle_concatenate(input_shapes, params)
        expected = (None, 10, 100)  # 20 + 30 + 50 = 100
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_concatenate_axis_1(self):
        """Test Concatenate along axis 1"""
        input_shapes = [
            (None, 5, 64),
            (None, 10, 64),
            (None, 15, 64)
        ]
        params = {"axis": 1}
        output_shape = handle_concatenate(input_shapes, params)
        expected = (None, 30, 64)  # 5 + 10 + 15 = 30
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_concatenate_with_none_dimensions(self):
        """Test Concatenate with None in concat axis"""
        input_shapes = [
            (None, None, 64),
            (None, 10, 64)
        ]
        params = {"axis": 1}
        output_shape = handle_concatenate(input_shapes, params)
        # When concat axis has None, output should be None
        expected = (None, None, 64)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_add_same_shapes(self):
        """Test Add layer with same shapes"""
        input_shapes = [
            (None, 28, 28, 64),
            (None, 28, 28, 64)
        ]
        params = {}
        output_shape = handle_add(input_shapes, params)
        expected = (None, 28, 28, 64)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
    
    def test_add_with_broadcasting(self):
        """Test Add layer with broadcasting (dimension = 1)"""
        input_shapes = [
            (None, 28, 28, 64),
            (None, 1, 1, 64)
        ]
        params = {}
        output_shape = handle_add(input_shapes, params)
        expected = (None, 28, 28, 64)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"


class TestNoneDimensionsHandling:
    """Test proper handling of None dimensions"""
    
    def test_conv2d_with_none_batch(self):
        """Test Conv2D with None batch dimension"""
        propagator = ShapePropagator()
        input_shape = (None, 28, 28, 3)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": (3, 3),
                "padding": "same",
                "stride": 1
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        expected = (None, 28, 28, 32)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
        assert output_shape[0] is None, "Batch dimension should remain None"
    
    def test_flatten_with_none_batch(self):
        """Test Flatten with None batch dimension"""
        propagator = ShapePropagator()
        input_shape = (None, 7, 7, 64)
        layer = {"type": "Flatten", "params": {}}
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        expected = (None, 7 * 7 * 64)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
        assert output_shape[0] is None, "Batch dimension should remain None"
    
    def test_dense_with_none_batch(self):
        """Test Dense with None batch dimension"""
        propagator = ShapePropagator()
        input_shape = (None, 128)
        layer = {"type": "Dense", "params": {"units": 64}}
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        expected = (None, 64)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
        assert output_shape[0] is None, "Batch dimension should remain None"
    
    def test_maxpooling2d_with_none_batch(self):
        """Test MaxPooling2D with None batch dimension"""
        propagator = ShapePropagator()
        input_shape = (None, 28, 28, 16)
        layer = {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": (2, 2),
                "stride": 2
            }
        }
        output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
        expected = (None, 14, 14, 16)
        assert output_shape == expected, f"Expected {expected}, got {output_shape}"
        assert output_shape[0] is None, "Batch dimension should remain None"


class TestFLOPsMemoryCalculations:
    """Test FLOPs and memory calculations"""
    
    def test_conv2d_flops_calculation(self):
        """Test FLOPs calculation for Conv2D"""
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 3)
        output_shape = (1, 26, 26, 16)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 16,
                "kernel_size": (3, 3)
            }
        }
        flops, memory, _, _ = propagator._compute_performance(layer, input_shape, output_shape)
        # FLOPs = 2 * kernel_h * kernel_w * input_channels * output_h * output_w * filters
        # FLOPs = 2 * 3 * 3 * 3 * 26 * 26 * 16
        expected_flops = 2 * 3 * 3 * 3 * 26 * 26 * 16
        assert flops == expected_flops, f"Expected {expected_flops} FLOPs, got {flops}"
        assert memory > 0, "Memory should be positive"
    
    def test_dense_flops_calculation(self):
        """Test FLOPs calculation for Dense"""
        propagator = ShapePropagator()
        input_shape = (1, 128)
        output_shape = (1, 64)
        layer = {
            "type": "Dense",
            "params": {
                "units": 64
            }
        }
        flops, memory, _, _ = propagator._compute_performance(layer, input_shape, output_shape)
        # FLOPs = 2 * input_features * output_features
        expected_flops = 2 * 128 * 64
        assert flops == expected_flops, f"Expected {expected_flops} FLOPs, got {flops}"
        assert memory > 0, "Memory should be positive"
    
    def test_lstm_flops_calculation(self):
        """Test FLOPs calculation for LSTM"""
        propagator = ShapePropagator()
        input_shape = (1, 10, 64)
        output_shape = (1, 10, 128)
        layer = {
            "type": "LSTM",
            "params": {
                "units": 128
            }
        }
        flops, memory, _, _ = propagator._compute_performance(layer, input_shape, output_shape)
        # FLOPs for LSTM = 4 * (input_size + hidden_size) * hidden_size * seq_len
        expected_flops = 4 * (64 + 128) * 128 * 10
        assert flops == expected_flops, f"Expected {expected_flops} FLOPs, got {flops}"
    
    def test_gru_flops_calculation(self):
        """Test FLOPs calculation for GRU"""
        propagator = ShapePropagator()
        input_shape = (1, 10, 64)
        output_shape = (1, 10, 128)
        layer = {
            "type": "GRU",
            "params": {
                "units": 128
            }
        }
        flops, memory, _, _ = propagator._compute_performance(layer, input_shape, output_shape)
        # FLOPs for GRU = 3 * (input_size + hidden_size) * hidden_size * seq_len
        expected_flops = 3 * (64 + 128) * 128 * 10
        assert flops == expected_flops, f"Expected {expected_flops} FLOPs, got {flops}"
    
    def test_memory_calculation_with_none(self):
        """Test memory calculation with None dimensions"""
        propagator = ShapePropagator()
        input_shape = (None, 28, 28, 3)
        output_shape = (None, 28, 28, 16)
        layer = {
            "type": "Conv2D",
            "params": {
                "filters": 16,
                "kernel_size": (3, 3)
            }
        }
        flops, memory, _, _ = propagator._compute_performance(layer, input_shape, output_shape)
        # None should be replaced with 1 for calculation
        # Memory = 1 * 28 * 28 * 16 * 4 bytes / (1024^2) MB
        expected_memory = 1 * 28 * 28 * 16 * 4 / (1024 ** 2)
        assert abs(memory - expected_memory) < 0.01, f"Expected ~{expected_memory} MB, got {memory} MB"


class TestComplexArchitectures:
    """Test complex multi-layer architectures"""
    
    def test_resnet_like_block(self):
        """Test ResNet-like block with skip connection"""
        propagator = ShapePropagator()
        input_shape = (None, 28, 28, 64)
        
        # Main path
        layers = [
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": 3, "padding": "same", "stride": 1}},
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": 3, "padding": "same", "stride": 1}},
        ]
        
        shape = input_shape
        for layer in layers:
            shape = propagator.propagate(shape, layer, framework="tensorflow")
        
        # After two conv layers with same padding, shape should be preserved
        expected = (None, 28, 28, 64)
        assert shape == expected, f"Expected {expected}, got {shape}"
    
    def test_encoder_decoder_architecture(self):
        """Test encoder-decoder architecture"""
        propagator = ShapePropagator()
        input_shape = (None, 10, 256)
        
        # Encoder: LSTM with return_sequences=True
        encoder_layer = {
            "type": "LSTM",
            "params": {
                "units": 128,
                "return_sequences": True
            }
        }
        encoder_output = propagator.propagate(input_shape, encoder_layer)
        assert encoder_output == (None, 10, 128)
        
        # Decoder: LSTM with return_sequences=False
        decoder_layer = {
            "type": "LSTM",
            "params": {
                "units": 256,
                "return_sequences": False
            }
        }
        decoder_output = propagator.propagate(encoder_output, decoder_layer)
        assert decoder_output == (None, 256)


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
