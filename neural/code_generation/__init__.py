"""
Code generation module for Neural DSL.

This module provides functionality to convert parsed Neural DSL models into
executable Python code for multiple deep learning frameworks.

Supported backends:
    - TensorFlow/Keras
    - PyTorch
    - ONNX

The code generator handles:
    - Layer instantiation and configuration
    - Shape propagation and validation
    - Optimizer and loss function setup
    - Training loop generation
    - Experiment tracking integration
"""

from neural.code_generation.base_generator import BaseCodeGenerator
from neural.code_generation.code_generator import (
    generate_code,
    generate_optimized_dsl,
    load_file,
    save_file,
    to_number,
)
from neural.code_generation.onnx_generator import ONNXGenerator, export_onnx
from neural.code_generation.pytorch_generator import PyTorchGenerator
from neural.code_generation.shape_policy_helpers import (
    ensure_2d_before_dense_pt,
    ensure_2d_before_dense_tf,
    get_rank_non_batch,
)
from neural.code_generation.tensorflow_generator import TensorFlowGenerator


try:
    from neural.code_generation.export import ModelExporter
except ImportError:
    ModelExporter = None

__all__ = [
    'generate_code',
    'save_file',
    'load_file',
    'generate_optimized_dsl',
    'to_number',
    'export_onnx',
    'TensorFlowGenerator',
    'PyTorchGenerator',
    'ONNXGenerator',
    'BaseCodeGenerator',
    'ensure_2d_before_dense_tf',
    'ensure_2d_before_dense_pt',
    'get_rank_non_batch',
    'ModelExporter'
]
