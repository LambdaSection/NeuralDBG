"""
Utility functions for Neural's shape propagation system.

This module provides helper functions for parameter extraction, shape validation,
and error detection used by the shape propagator.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union, Callable

def extract_param(params: Dict[str, Any],
                 key: str,
                 default: Any = None,
                 transform: Optional[Callable] = None) -> Any:
    """Extract parameter with HPO handling and optional transformation.

    Args:
        params: Dictionary of parameters
        key: Parameter key to extract
        default: Default value if parameter is not found
        transform: Optional function to transform the parameter value

    Returns:
        The extracted parameter value
    """
    value = params.get(key, default)

    # Handle HPO dictionary format
    if isinstance(value, dict):
        if 'value' in value:
            value = value['value']
        else:
            value = default

    # Apply transformation if provided
    if transform and value is not None:
        value = transform(value)

    return value

def calculate_output_dims(input_dims: Tuple[int, ...],
                         kernel_size: Tuple[int, ...],
                         stride: Tuple[int, ...],
                         padding: Union[str, int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Calculate output dimensions for convolution or pooling operations.

    Args:
        input_dims: Input spatial dimensions
        kernel_size: Kernel size for each dimension
        stride: Stride for each dimension
        padding: Padding mode ('valid', 'same') or explicit padding

    Returns:
        Output spatial dimensions
    """
    output_dims = []

    for i, dim in enumerate(input_dims):
        k = kernel_size[i] if i < len(kernel_size) else kernel_size[0]
        s = stride[i] if i < len(stride) else stride[0]

        if padding == 'valid':
            p = 0
        elif padding == 'same':
            p = (k - 1) // 2
        elif isinstance(padding, (int, float)):
            p = int(padding)
        elif isinstance(padding, (tuple, list)):
            p = padding[i] if i < len(padding) else padding[0]
        else:
            p = 0

        output_dim = (dim + 2 * p - k) // s + 1
        output_dims.append(output_dim)

    return tuple(output_dims)

def detect_shape_issues(shape_history: List[Tuple[str, Tuple[int, ...]]]) -> List[Dict[str, Any]]:
    """Detect potential issues in shape propagation.

    Args:
        shape_history: List of (layer_name, output_shape) tuples

    Returns:
        List of detected issues with type, message, and layer index
    """
    issues = []

    # Check for extreme tensor size changes
    for i in range(1, len(shape_history)):
        prev_layer, prev_shape = shape_history[i-1]
        curr_layer, curr_shape = shape_history[i]

        prev_size = np.prod([dim for dim in prev_shape if dim is not None])
        curr_size = np.prod([dim for dim in curr_shape if dim is not None])

        if curr_size > prev_size * 10:
            issues.append({
                'type': 'warning',
                'message': f'Large tensor size increase at {curr_layer}: {prev_size} → {curr_size}',
                'layer_index': i
            })
        elif curr_size * 100 < prev_size and prev_size > 1000:
            issues.append({
                'type': 'info',
                'message': f'Significant tensor size reduction at {curr_layer}: {prev_size} → {curr_size}',
                'layer_index': i
            })

    # Check for very large tensors
    for i, (layer_name, shape) in enumerate(shape_history):
        size = np.prod([dim for dim in shape if dim is not None])
        memory_mb = size * 4 / (1024 * 1024)  # Assuming float32

        if memory_mb > 1000:
            issues.append({
                'type': 'warning',
                'message': f'Very large tensor at {layer_name}: {shape} ({memory_mb:.2f} MB)',
                'layer_index': i
            })

    # Check for potential bottlenecks
    for i in range(1, len(shape_history) - 1):
        prev_size = np.prod([dim for dim in shape_history[i-1][1] if dim is not None])
        curr_size = np.prod([dim for dim in shape_history[i][1] if dim is not None])
        next_size = np.prod([dim for dim in shape_history[i+1][1] if dim is not None])

        if curr_size < prev_size * 0.1 and curr_size < next_size * 0.1:
            issues.append({
                'type': 'bottleneck',
                'message': f'Potential information bottleneck at {shape_history[i][0]}: {curr_size} elements',
                'layer_index': i
            })

    return issues

def suggest_optimizations(shape_history: List[Tuple[str, Tuple[int, ...]]]) -> List[Dict[str, Any]]:
    """Suggest optimizations based on shape analysis.

    Args:
        shape_history: List of (layer_name, output_shape) tuples

    Returns:
        List of optimization suggestions with type, message, and layer index
    """
    suggestions = []

    # Look for opportunities to reduce dimensions
    for i, (layer_name, shape) in enumerate(shape_history):
        if len(shape) == 4:  # Conv layers
            if shape[1] > 100 and shape[2] > 100 and shape[3] > 64:
                suggestions.append({
                    'type': 'optimization',
                    'message': f'Consider adding pooling after {layer_name} to reduce spatial dimensions {shape[1]}x{shape[2]}',
                    'layer_index': i
                })

    # Check for potential over-parameterization
    for i, (layer_name, shape) in enumerate(shape_history):
        if 'Dense' in layer_name and len(shape) == 2:
            if shape[1] > 4096:
                suggestions.append({
                    'type': 'optimization',
                    'message': f'Consider reducing units in {layer_name} ({shape[1]} units may be excessive)',
                    'layer_index': i
                })

    return suggestions

def format_error_message(error_type: str, details: Dict[str, Any]) -> str:
    """Format user-friendly error messages.

    Args:
        error_type: Type of error
        details: Dictionary with error details

    Returns:
        Formatted error message
    """
    messages = {
        'invalid_input_shape': f"Invalid input shape: {details['shape']}. Expected a tuple with positive dimensions.",
        'kernel_too_large': f"Kernel size {details['kernel_size']} is too large for input dimensions {details['input_dims']}.",
        'missing_parameter': f"Missing required parameter '{details['param']}' for {details['layer_type']} layer.",
        'incompatible_shapes': f"Incompatible shapes: cannot connect {details['from_shape']} to {details['to_shape']}.",
        'negative_stride': f"Stride must be positive, got {details['stride']} for {details['layer_type']} layer.",
        'negative_filters': f"Filters must be positive, got {details['filters']} for Conv2D layer."
    }

    return messages.get(error_type, f"Error: {details}")

def calculate_memory_usage(shape: Tuple[int, ...], dtype: str = 'float32') -> int:
    """Calculate memory usage for a tensor with given shape and dtype.

    Args:
        shape: Tensor shape
        dtype: Data type of tensor elements

    Returns:
        Memory usage in bytes
    """
    bytes_per_element = {
        'float32': 4,
        'float16': 2,
        'int32': 4,
        'int64': 8,
        'uint8': 1,
        'bool': 1
    }.get(dtype, 4)

    num_elements = np.prod([dim for dim in shape if dim is not None])
    return int(num_elements * bytes_per_element)

def format_memory_size(bytes: int) -> str:
    """Format memory size in human-readable form.

    Args:
        bytes: Memory size in bytes

    Returns:
        Formatted memory size string
    """
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.2f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes / (1024 * 1024 * 1024):.2f} GB"
