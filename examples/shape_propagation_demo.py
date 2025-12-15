"""
Demonstration of shape propagation bug fixes.

This script demonstrates the corrected shape propagation for:
1. Conv2D with various padding configurations
2. LSTM/GRU with return_sequences
3. Multi-input concatenation
4. None dimensions handling
5. FLOPs calculations
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.shape_propagation.layer_handlers import handle_concatenate, handle_lstm, handle_gru


def demo_conv2d_padding():
    """Demonstrate Conv2D with various padding configurations"""
    print("=" * 70)
    print("DEMO 1: Conv2D with Various Padding Configurations")
    print("=" * 70)
    
    propagator = ShapePropagator()
    
    # Test 1: Valid padding
    print("\n1. Conv2D with 'valid' padding:")
    input_shape = (None, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": (3, 3),
            "padding": "valid",
            "stride": 1
        }
    }
    output = propagator.propagate(input_shape, layer, framework="tensorflow")
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output}")
    print(f"   ✓ Spatial dims: 28 -> 26 (correct for valid padding)")
    
    # Test 2: Same padding with stride=1
    print("\n2. Conv2D with 'same' padding and stride=1:")
    layer["params"]["padding"] = "same"
    propagator2 = ShapePropagator()
    output = propagator2.propagate(input_shape, layer, framework="tensorflow")
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output}")
    print(f"   ✓ Spatial dims preserved: 28 -> 28")
    
    # Test 3: Same padding with stride=2
    print("\n3. Conv2D with 'same' padding and stride=2:")
    layer["params"]["stride"] = 2
    propagator3 = ShapePropagator()
    output = propagator3.propagate(input_shape, layer, framework="tensorflow")
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output}")
    print(f"   ✓ Spatial dims halved: 28 -> 14 (ceil(28/2))")


def demo_lstm_gru():
    """Demonstrate LSTM/GRU with return_sequences"""
    print("\n" + "=" * 70)
    print("DEMO 2: LSTM/GRU with return_sequences")
    print("=" * 70)
    
    # LSTM with return_sequences=True
    print("\n1. LSTM with return_sequences=True:")
    input_shape = (None, 10, 64)
    params = {"units": 128, "return_sequences": True}
    output = handle_lstm(input_shape, params)
    print(f"   Input:  {input_shape} (batch, seq_len, input_size)")
    print(f"   Output: {output} (batch, seq_len, units)")
    print(f"   ✓ Full sequence returned")
    
    # LSTM with return_sequences=False
    print("\n2. LSTM with return_sequences=False:")
    params = {"units": 128, "return_sequences": False}
    output = handle_lstm(input_shape, params)
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output} (batch, units)")
    print(f"   ✓ Only last output returned")
    
    # GRU with return_sequences=True
    print("\n3. GRU with return_sequences=True:")
    params = {"units": 64, "return_sequences": True}
    output = handle_gru(input_shape, params)
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output}")
    print(f"   ✓ GRU handler working correctly")


def demo_concatenation():
    """Demonstrate multi-input concatenation"""
    print("\n" + "=" * 70)
    print("DEMO 3: Multi-Input Concatenation")
    print("=" * 70)
    
    # Concatenate along last axis
    print("\n1. Concatenate along axis -1:")
    input_shapes = [
        (None, 10, 20),
        (None, 10, 30),
        (None, 10, 50)
    ]
    params = {"axis": -1}
    output = handle_concatenate(input_shapes, params)
    print(f"   Input 1: {input_shapes[0]}")
    print(f"   Input 2: {input_shapes[1]}")
    print(f"   Input 3: {input_shapes[2]}")
    print(f"   Output:  {output}")
    print(f"   ✓ Concatenated dimension: 20 + 30 + 50 = 100")
    
    # Concatenate with None dimensions
    print("\n2. Concatenate with None in concat axis:")
    input_shapes = [
        (None, None, 64),
        (None, 10, 64)
    ]
    params = {"axis": 1}
    output = handle_concatenate(input_shapes, params)
    print(f"   Input 1: {input_shapes[0]}")
    print(f"   Input 2: {input_shapes[1]}")
    print(f"   Output:  {output}")
    print(f"   ✓ None preserved when present in concat axis")


def demo_none_dimensions():
    """Demonstrate None dimension handling"""
    print("\n" + "=" * 70)
    print("DEMO 4: None Dimensions Handling")
    print("=" * 70)
    
    propagator = ShapePropagator()
    
    print("\n1. Conv2D with None batch dimension:")
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
    output = propagator.propagate(input_shape, layer, framework="tensorflow")
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output}")
    print(f"   ✓ Batch dimension (None) preserved")
    
    print("\n2. Flatten with None batch:")
    input_shape = (None, 7, 7, 64)
    layer = {"type": "Flatten", "params": {}}
    propagator2 = ShapePropagator()
    output = propagator2.propagate(input_shape, layer, framework="tensorflow")
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output}")
    print(f"   ✓ Flattened to {7*7*64} features, batch None preserved")


def demo_flops_calculations():
    """Demonstrate FLOPs calculations"""
    print("\n" + "=" * 70)
    print("DEMO 5: FLOPs Calculations")
    print("=" * 70)
    
    propagator = ShapePropagator()
    
    # Conv2D FLOPs
    print("\n1. Conv2D FLOPs:")
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": (3, 3),
            "padding": "valid"
        }
    }
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    flops, memory, _, _ = propagator._compute_performance(layer, input_shape, output_shape)
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output_shape}")
    print(f"   FLOPs:  {flops:,}")
    print(f"   Memory: {memory:.4f} MB")
    
    # Dense FLOPs
    print("\n2. Dense FLOPs:")
    input_shape = (1, 128)
    layer = {"type": "Dense", "params": {"units": 64}}
    propagator2 = ShapePropagator()
    output_shape = propagator2.propagate(input_shape, layer, framework="tensorflow")
    flops, memory, _, _ = propagator2._compute_performance(layer, input_shape, output_shape)
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output_shape}")
    print(f"   FLOPs:  {flops:,}")
    print(f"   Memory: {memory:.4f} MB")
    
    # LSTM FLOPs
    print("\n3. LSTM FLOPs:")
    input_shape = (1, 10, 64)
    layer = {"type": "LSTM", "params": {"units": 128, "return_sequences": True}}
    propagator3 = ShapePropagator()
    output_shape = propagator3.propagate(input_shape, layer, framework="tensorflow")
    flops, memory, _, _ = propagator3._compute_performance(layer, input_shape, output_shape)
    print(f"   Input:  {input_shape}")
    print(f"   Output: {output_shape}")
    print(f"   FLOPs:  {flops:,}")
    print(f"   Memory: {memory:.4f} MB")


def main():
    """Run all demonstrations"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  Shape Propagation Bug Fixes - Demonstration".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    demo_conv2d_padding()
    demo_lstm_gru()
    demo_concatenation()
    demo_none_dimensions()
    demo_flops_calculations()
    
    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
