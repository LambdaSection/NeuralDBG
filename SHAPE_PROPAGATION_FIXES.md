# Shape Propagation Bug Fixes

## Overview
This document describes the comprehensive bug fixes implemented in the shape propagation system for the Neural DSL framework.

## Bugs Fixed

### 1. Conv2D Padding Configuration Issues

**Problem:** Conv2D layers with various padding configurations (same, valid, explicit) were not calculating output shapes correctly, especially with stride > 1.

**Fixes:**
- Improved `_calculate_padding()` method to properly handle:
  - `'same'` padding with stride > 1 (now correctly calculates ceil(input/stride))
  - `'valid'` padding (returns 0)
  - Explicit integer padding
  - Tuple/list padding for asymmetric cases
  - Dictionary parameter handling from HPO
- Updated `_handle_conv2d()` to:
  - Normalize stride to tuple format
  - Handle None dimensions properly
  - Ensure minimum output dimension of 1
  - Support both channels_first and channels_last data formats

**Test Coverage:**
- `test_conv2d_valid_padding()`
- `test_conv2d_same_padding_stride1()`
- `test_conv2d_same_padding_stride2()`
- `test_conv2d_explicit_padding()`
- `test_conv2d_asymmetric_kernel_same_padding()`

### 2. LSTM/GRU return_sequences Shape Handling

**Problem:** LSTM and GRU layers did not properly handle the `return_sequences` parameter, leading to incorrect output shapes.

**Fixes:**
- Enhanced `handle_lstm()` in `layer_handlers.py`:
  - When `return_sequences=True`: Returns (batch, seq_len, units)
  - When `return_sequences=False`: Returns (batch, units)
- Added new `handle_gru()` function with same behavior as LSTM
- Added `_handle_gru()` internal handler in shape_propagator.py
- Updated param_aliases to include GRU mappings

**Test Coverage:**
- `test_lstm_return_sequences_true()`
- `test_lstm_return_sequences_false()`
- `test_gru_return_sequences_true()`
- `test_gru_return_sequences_false()`
- `test_lstm_with_batch_size()`

### 3. Transformer Layer Shape Propagation

**Problem:** Transformer layers (TransformerEncoder, TransformerDecoder, MultiHeadAttention) had incomplete or incorrect shape propagation.

**Fixes:**
- TransformerEncoder now properly handles framework differences:
  - TensorFlow: Preserves full shape (batch, seq_len, d_model)
  - PyTorch: Returns (batch, seq_len)
- TransformerDecoder follows same pattern as encoder
- MultiHeadAttention preserves input shape: (batch, seq_len, d_model)
- Improved `_handle_multiheadattention()` method

**Test Coverage:**
- `test_transformer_encoder_tensorflow()`
- `test_transformer_encoder_pytorch()`
- `test_multihead_attention()`

### 4. Multi-Input and Concatenation Layer Shapes

**Problem:** Concatenate and Add layers did not properly handle multiple inputs, especially with None dimensions.

**Fixes:**
- Enhanced `handle_concatenate()`:
  - Properly handles None (dynamic) dimensions
  - Validates shape compatibility across all inputs
  - If concat axis contains None in any input, output is None
  - Sums dimensions along concatenation axis
- Improved `handle_add()`:
  - Supports broadcasting (dimension = 1)
  - Handles None dimensions properly
  - Validates shape compatibility
- Updated `propagate_model()` to properly handle multi-input layers

**Test Coverage:**
- `test_concatenate_axis_last()`
- `test_concatenate_axis_1()`
- `test_concatenate_with_none_dimensions()`
- `test_add_same_shapes()`
- `test_add_with_broadcasting()`

### 5. None Dimensions Handling

**Problem:** None dimensions (representing dynamic batch sizes or unknown dimensions) were not consistently handled throughout the system.

**Fixes:**
- Updated `calculate_output_dims()` in `utils.py`:
  - Returns None for dimensions that are None in input
  - Preserves None through calculations
  - Ensures minimum output of 1 for concrete dimensions
- Enhanced all layer handlers to preserve None dimensions:
  - `_handle_conv2d()`: Handles None in spatial dimensions
  - `_handle_maxpooling2d()`: Preserves None in calculations
  - `_handle_flatten()`: Excludes None from product calculations
  - `_handle_upsampling2d()`: Handles None in multiplication
  - Zero padding and cropping layers: Handle None in arithmetic
- Updated `_compute_performance()` to replace None with 1 for FLOPs/memory calculations

**Test Coverage:**
- `test_conv2d_with_none_batch()`
- `test_flatten_with_none_batch()`
- `test_dense_with_none_batch()`
- `test_maxpooling2d_with_none_batch()`
- `test_memory_calculation_with_none()`

### 6. FLOPs and Memory Calculations

**Problem:** FLOPs and memory calculations were incomplete or incorrect for many layer types.

**Fixes:**
- Comprehensive `_compute_performance()` method updates:
  - **Conv2D**: FLOPs = 2 × kernel_h × kernel_w × input_channels × output_h × output_w × filters
  - **Dense**: FLOPs = 2 × input_features × output_features
  - **LSTM**: FLOPs = 4 × (input_size + hidden_size) × hidden_size × seq_len
  - **GRU**: FLOPs = 3 × (input_size + hidden_size) × hidden_size × seq_len
  - **Pooling layers**: FLOPs = output_tensor_size (minimal cost)
  - **BatchNorm/Dropout**: FLOPs = output_tensor_size (minimal cost)
- Memory calculation properly handles None dimensions
- Memory usage in MB: tensor_size × 4 bytes (float32) / (1024²)
- Added compute_time and transfer_time estimates

**Test Coverage:**
- `test_conv2d_flops_calculation()`
- `test_dense_flops_calculation()`
- `test_lstm_flops_calculation()`
- `test_gru_flops_calculation()`
- `test_memory_calculation_with_none()`

## Additional Improvements

### Better Error Handling
- All handlers properly validate input shapes
- Dictionary parameter handling (from HPO) throughout
- Graceful degradation with debug logging

### Code Quality
- Consistent None handling across all functions
- Improved type hints and documentation
- Better separation of concerns
- Comprehensive debug logging

### Performance Optimizations
- Efficient calculations avoiding unnecessary operations
- Proper handling of edge cases
- Optimized memory footprint calculations

## Testing

A comprehensive test suite has been created in `tests/test_shape_propagation_fixes.py` with 6 test classes covering:
1. Conv2D padding configurations (6 tests)
2. LSTM/GRU return_sequences (5 tests)
3. Transformer layers (3 tests)
4. Multi-input and concatenation (5 tests)
5. None dimensions handling (4 tests)
6. FLOPs and memory calculations (5 tests)
7. Complex architectures (2 tests)

Total: 30 targeted tests demonstrating all fixes

## Files Modified

1. **neural/shape_propagation/shape_propagator.py**
   - Enhanced Conv2D padding calculation
   - Added GRU handler
   - Improved None dimension handling throughout
   - Fixed FLOPs/memory calculations for all layer types
   - Better multi-input support

2. **neural/shape_propagation/layer_handlers.py**
   - Added `handle_gru()` function
   - Enhanced `handle_concatenate()` with None support
   - Improved `handle_add()` with broadcasting
   - Updated `handle_lstm()` for clarity
   - Fixed None handling in padding/cropping layers

3. **neural/shape_propagation/utils.py**
   - Updated `calculate_output_dims()` to handle None
   - Improved 'same' padding calculation for stride > 1
   - Added safety checks for minimum dimensions

4. **tests/test_shape_propagation_fixes.py** (NEW)
   - Comprehensive test suite for all fixes

## Backward Compatibility

All fixes maintain backward compatibility with existing code:
- Default behaviors unchanged when parameters are standard
- None handling is additive (doesn't break concrete dimensions)
- Dictionary parameter support is additive
- All existing tests should continue to pass

## Usage Examples

### Conv2D with various padding
```python
propagator = ShapePropagator()

# Same padding with stride > 1
layer = {
    "type": "Conv2D",
    "params": {
        "filters": 32,
        "kernel_size": 3,
        "padding": "same",
        "stride": 2
    }
}
# (None, 28, 28, 3) -> (None, 14, 14, 32)
```

### LSTM with return_sequences
```python
# Return full sequence
layer = {
    "type": "LSTM",
    "params": {
        "units": 128,
        "return_sequences": True
    }
}
# (None, 10, 64) -> (None, 10, 128)

# Return last output only
layer = {
    "type": "LSTM",
    "params": {
        "units": 128,
        "return_sequences": False
    }
}
# (None, 10, 64) -> (None, 128)
```

### Concatenate with None dimensions
```python
from neural.shape_propagation.layer_handlers import handle_concatenate

input_shapes = [
    (None, 10, 20),
    (None, 10, 30)
]
params = {"axis": -1}
output = handle_concatenate(input_shapes, params)
# Output: (None, 10, 50)
```

## Conclusion

These comprehensive fixes address all major shape propagation issues in the Neural DSL framework, providing:
- Correct shape calculations for all layer types
- Proper handling of dynamic (None) dimensions
- Accurate FLOPs and memory estimates
- Support for complex multi-input architectures
- Robust error handling and validation

The system is now production-ready for a wide variety of neural network architectures.
