"""
NAS-specific operations and layer primitives.

Provides building blocks for neural architecture search.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NASOperation(ABC):
    """Base class for NAS operations."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def to_layer_config(self, in_channels: int, out_channels: int) -> Dict[str, Any]:
        """Convert operation to layer configuration."""
        pass
    
    @abstractmethod
    def get_parameter_count(self, in_channels: int, out_channels: int) -> int:
        """Estimate number of parameters."""
        pass


class SkipConnection(NASOperation):
    """Identity/skip connection."""
    
    def __init__(self):
        super().__init__('skip_connect')
    
    def to_layer_config(self, in_channels: int, out_channels: int) -> Dict[str, Any]:
        """Convert to layer config (identity)."""
        if in_channels == out_channels:
            return {'type': 'Identity', 'params': {}}
        else:
            return {
                'type': 'Conv2D',
                'params': {
                    'filters': out_channels,
                    'kernel_size': 1,
                    'strides': 1,
                    'padding': 'same'
                }
            }
    
    def get_parameter_count(self, in_channels: int, out_channels: int) -> int:
        """Calculate parameters."""
        if in_channels == out_channels:
            return 0
        return in_channels * out_channels


class FactorizedReduce(NASOperation):
    """Factorized reduction operation."""
    
    def __init__(self):
        super().__init__('factorized_reduce')
    
    def to_layer_config(self, in_channels: int, out_channels: int) -> Dict[str, Any]:
        """Convert to layer config (strided convolution)."""
        return {
            'type': 'Conv2D',
            'params': {
                'filters': out_channels,
                'kernel_size': 1,
                'strides': 2,
                'padding': 'valid'
            }
        }
    
    def get_parameter_count(self, in_channels: int, out_channels: int) -> int:
        """Calculate parameters."""
        return in_channels * out_channels


class SepConv(NASOperation):
    """Separable convolution."""
    
    def __init__(self, kernel_size: int = 3):
        super().__init__(f'sep_conv_{kernel_size}x{kernel_size}')
        self.kernel_size = kernel_size
    
    def to_layer_config(self, in_channels: int, out_channels: int) -> Dict[str, Any]:
        """Convert to layer config (depthwise + pointwise)."""
        return {
            'type': 'SeparableConv2D',
            'params': {
                'filters': out_channels,
                'kernel_size': self.kernel_size,
                'padding': 'same'
            }
        }
    
    def get_parameter_count(self, in_channels: int, out_channels: int) -> int:
        """Calculate parameters."""
        depthwise = in_channels * self.kernel_size * self.kernel_size
        pointwise = in_channels * out_channels
        return depthwise + pointwise


class DilatedConv(NASOperation):
    """Dilated (atrous) convolution."""
    
    def __init__(self, kernel_size: int = 3, dilation_rate: int = 2):
        super().__init__(f'dil_conv_{kernel_size}x{kernel_size}_d{dilation_rate}')
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
    
    def to_layer_config(self, in_channels: int, out_channels: int) -> Dict[str, Any]:
        """Convert to layer config."""
        return {
            'type': 'Conv2D',
            'params': {
                'filters': out_channels,
                'kernel_size': self.kernel_size,
                'dilation_rate': self.dilation_rate,
                'padding': 'same'
            }
        }
    
    def get_parameter_count(self, in_channels: int, out_channels: int) -> int:
        """Calculate parameters."""
        return in_channels * out_channels * self.kernel_size * self.kernel_size


class PoolBN(NASOperation):
    """Pooling followed by batch normalization."""
    
    def __init__(self, pool_type: str = 'max', pool_size: int = 3):
        super().__init__(f'{pool_type}_pool_{pool_size}x{pool_size}_bn')
        self.pool_type = pool_type
        self.pool_size = pool_size
    
    def to_layer_config(self, in_channels: int, out_channels: int) -> Dict[str, Any]:
        """Convert to layer config (pooling + BN)."""
        pool_layer = 'MaxPooling2D' if self.pool_type == 'max' else 'AveragePooling2D'
        
        return {
            'type': 'Sequential',
            'layers': [
                {
                    'type': pool_layer,
                    'params': {
                        'pool_size': self.pool_size,
                        'strides': 1,
                        'padding': 'same'
                    }
                },
                {
                    'type': 'BatchNormalization',
                    'params': {}
                }
            ]
        }
    
    def get_parameter_count(self, in_channels: int, out_channels: int) -> int:
        """Calculate parameters (only BN has parameters)."""
        return 4 * in_channels


class ZeroOperation(NASOperation):
    """Zero operation (removes connection)."""
    
    def __init__(self):
        super().__init__('zero')
    
    def to_layer_config(self, in_channels: int, out_channels: int) -> Dict[str, Any]:
        """Convert to layer config (zero)."""
        return {'type': 'Zero', 'params': {}}
    
    def get_parameter_count(self, in_channels: int, out_channels: int) -> int:
        """Zero parameters."""
        return 0


class InvertedResidual(NASOperation):
    """Inverted residual block (MobileNetV2 style)."""
    
    def __init__(self, expand_ratio: int = 6, kernel_size: int = 3):
        super().__init__(f'inverted_residual_e{expand_ratio}_k{kernel_size}')
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
    
    def to_layer_config(self, in_channels: int, out_channels: int) -> Dict[str, Any]:
        """Convert to layer config."""
        expanded_channels = in_channels * self.expand_ratio
        
        return {
            'type': 'Sequential',
            'layers': [
                {
                    'type': 'Conv2D',
                    'params': {
                        'filters': expanded_channels,
                        'kernel_size': 1,
                        'padding': 'same'
                    }
                },
                {
                    'type': 'BatchNormalization',
                    'params': {}
                },
                {
                    'type': 'ReLU',
                    'params': {}
                },
                {
                    'type': 'DepthwiseConv2D',
                    'params': {
                        'kernel_size': self.kernel_size,
                        'padding': 'same'
                    }
                },
                {
                    'type': 'BatchNormalization',
                    'params': {}
                },
                {
                    'type': 'ReLU',
                    'params': {}
                },
                {
                    'type': 'Conv2D',
                    'params': {
                        'filters': out_channels,
                        'kernel_size': 1,
                        'padding': 'same'
                    }
                },
                {
                    'type': 'BatchNormalization',
                    'params': {}
                }
            ]
        }
    
    def get_parameter_count(self, in_channels: int, out_channels: int) -> int:
        """Calculate parameters."""
        expanded = in_channels * self.expand_ratio
        expand_params = in_channels * expanded
        depthwise_params = expanded * self.kernel_size * self.kernel_size
        project_params = expanded * out_channels
        bn_params = 4 * (expanded + expanded + out_channels)
        
        return expand_params + depthwise_params + project_params + bn_params


def get_nas_primitives() -> List[NASOperation]:
    """Get all standard NAS operations."""
    return [
        SkipConnection(),
        SepConv(3),
        SepConv(5),
        DilatedConv(3, 2),
        DilatedConv(5, 2),
        PoolBN('max', 3),
        PoolBN('avg', 3),
        InvertedResidual(3, 3),
        InvertedResidual(6, 3),
        ZeroOperation(),
    ]


def create_nas_cell(
    operations: List[NASOperation],
    num_nodes: int = 4,
    in_channels: int = 64,
    out_channels: int = 64
) -> Dict[str, Any]:
    """
    Create a NAS cell with multiple nodes and operations.
    
    Args:
        operations: List of operations to use
        num_nodes: Number of intermediate nodes
        in_channels: Input channels
        out_channels: Output channels
    
    Returns:
        Cell configuration dictionary
    """
    cell_config = {
        'type': 'NASCell',
        'num_nodes': num_nodes,
        'operations': [],
        'connections': []
    }
    
    for node_idx in range(num_nodes):
        node_ops = []
        
        for prev_idx in range(node_idx + 1):
            op = np.random.choice(operations)
            node_ops.append({
                'from_node': prev_idx,
                'to_node': node_idx + 1,
                'operation': op.to_layer_config(in_channels, out_channels)
            })
        
        cell_config['operations'].extend(node_ops)
    
    return cell_config


def estimate_model_size(architecture: Dict[str, Any]) -> int:
    """
    Estimate total parameter count of an architecture.
    
    Args:
        architecture: Architecture configuration
    
    Returns:
        Estimated parameter count
    """
    total_params = 0
    
    for layer in architecture.get('layers', []):
        layer_type = layer.get('type', '')
        params = layer.get('params', {})
        
        if layer_type == 'Conv2D':
            in_ch = params.get('in_channels', 64)
            out_ch = params.get('filters', 64)
            kernel = params.get('kernel_size', 3)
            total_params += in_ch * out_ch * kernel * kernel
        
        elif layer_type == 'Dense':
            in_feat = params.get('in_features', 512)
            out_feat = params.get('units', 512)
            total_params += in_feat * out_feat
        
        elif layer_type == 'BatchNormalization':
            channels = params.get('num_features', 64)
            total_params += 4 * channels
    
    return total_params


def compute_flops(architecture: Dict[str, Any], input_shape: Tuple[int, ...]) -> int:
    """
    Estimate FLOPs for an architecture.
    
    Args:
        architecture: Architecture configuration
        input_shape: Input tensor shape (H, W, C)
    
    Returns:
        Estimated FLOPs
    """
    total_flops = 0
    current_shape = input_shape
    
    for layer in architecture.get('layers', []):
        layer_type = layer.get('type', '')
        params = layer.get('params', {})
        
        if layer_type == 'Conv2D':
            H, W, C_in = current_shape
            C_out = params.get('filters', 64)
            K = params.get('kernel_size', 3)
            stride = params.get('strides', 1)
            
            H_out = (H + stride - 1) // stride
            W_out = (W + stride - 1) // stride
            
            flops = H_out * W_out * C_out * C_in * K * K
            total_flops += flops
            
            current_shape = (H_out, W_out, C_out)
        
        elif layer_type == 'Dense':
            in_feat = params.get('in_features', 512)
            out_feat = params.get('units', 512)
            
            flops = in_feat * out_feat
            total_flops += flops
    
    return total_flops
