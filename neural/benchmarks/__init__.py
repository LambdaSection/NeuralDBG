"""
Neural DSL Benchmarking Suite

This module provides comprehensive benchmarking tools for comparing Neural DSL
against popular ML frameworks including Keras, PyTorch Lightning, Fast.ai, and Ludwig.
"""

from .benchmark_runner import BenchmarkRunner, BenchmarkResult
from .framework_implementations import (
    FastAIImplementation,
    KerasImplementation,
    LudwigImplementation,
    NeuralDSLImplementation,
    PyTorchLightningImplementation,
)
from .metrics_collector import MetricsCollector
from .report_generator import ReportGenerator

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "KerasImplementation",
    "PyTorchLightningImplementation",
    "FastAIImplementation",
    "LudwigImplementation",
    "NeuralDSLImplementation",
    "MetricsCollector",
    "ReportGenerator",
]
