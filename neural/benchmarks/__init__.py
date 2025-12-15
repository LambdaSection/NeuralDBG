"""
Neural DSL Benchmarking Suite

This module provides comprehensive benchmarking tools for comparing Neural DSL
against popular ML frameworks including Raw TensorFlow, Raw PyTorch, Keras, 
PyTorch Lightning, Fast.ai, and Ludwig.

Key Features:
- Fair, reproducible comparisons
- Multiple metrics: LOC, dev time, training time, accuracy
- Publication-quality visualizations
- Automated report generation
- Multi-backend support

Quick Start:
    >>> from neural.benchmarks import BenchmarkRunner, NeuralDSLImplementation, KerasImplementation
    >>> runner = BenchmarkRunner()
    >>> frameworks = [NeuralDSLImplementation(), KerasImplementation()]
    >>> tasks = [{"name": "MNIST", "dataset": "mnist", "epochs": 5, "batch_size": 32}]
    >>> results = runner.run_all_benchmarks(frameworks, tasks)
"""

from .benchmark_runner import BenchmarkResult, BenchmarkRunner
from .framework_implementations import (
    FastAIImplementation,
    FrameworkImplementation,
    KerasImplementation,
    LudwigImplementation,
    NeuralDSLImplementation,
    PyTorchLightningImplementation,
    RawPyTorchImplementation,
    RawTensorFlowImplementation,
)
from .metrics_collector import (
    CodeMetrics,
    MetricsCollector,
    PerformanceTimer,
    ResourceMonitor,
    SystemInfo,
)
from .report_generator import ReportGenerator


try:
    from .visualization import BenchmarkVisualizer
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False
    BenchmarkVisualizer = None

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "BenchmarkRunner",
    "BenchmarkResult",
    "ReportGenerator",
    
    # Framework implementations
    "FrameworkImplementation",
    "NeuralDSLImplementation",
    "KerasImplementation",
    "RawTensorFlowImplementation",
    "PyTorchLightningImplementation",
    "RawPyTorchImplementation",
    "FastAIImplementation",
    "LudwigImplementation",
    
    # Metrics collection
    "MetricsCollector",
    "ResourceMonitor",
    "PerformanceTimer",
    "CodeMetrics",
    "SystemInfo",
    
    # Visualization (optional)
    "BenchmarkVisualizer",
]

# Module-level convenience functions

def quick_benchmark(frameworks=None, epochs=3, batch_size=32):
    """
    Run a quick benchmark with minimal configuration.
    
    Args:
        frameworks: List of framework names (default: ["neural", "keras"])
        epochs: Number of training epochs (default: 3)
        batch_size: Training batch size (default: 32)
    
    Returns:
        List of BenchmarkResult objects
    
    Example:
        >>> results = quick_benchmark(frameworks=["neural", "keras"])
        >>> for r in results:
        ...     print(f"{r.framework}: {r.lines_of_code} LOC, {r.model_accuracy:.4f} accuracy")
    """
    if frameworks is None:
        frameworks = ["neural", "keras"]
    
    framework_map = {
        "neural": NeuralDSLImplementation,
        "keras": KerasImplementation,
        "raw-tensorflow": RawTensorFlowImplementation,
        "pytorch-lightning": PyTorchLightningImplementation,
        "raw-pytorch": RawPyTorchImplementation,
        "fastai": FastAIImplementation,
        "ludwig": LudwigImplementation,
    }
    
    framework_instances = []
    for fw_name in frameworks:
        if fw_name in framework_map:
            try:
                framework_instances.append(framework_map[fw_name]())
            except ImportError:
                print(f"Warning: {fw_name} not available, skipping...")
    
    if not framework_instances:
        raise ValueError("No valid frameworks available")
    
    tasks = [{
        "name": "QuickBenchmark",
        "dataset": "mnist",
        "epochs": epochs,
        "batch_size": batch_size,
    }]
    
    runner = BenchmarkRunner(verbose=True)
    return runner.run_all_benchmarks(framework_instances, tasks)


def compare_frameworks(framework_a, framework_b, epochs=5, batch_size=32):
    """
    Compare two frameworks head-to-head.
    
    Args:
        framework_a: First framework name (e.g., "neural")
        framework_b: Second framework name (e.g., "keras")
        epochs: Number of training epochs (default: 5)
        batch_size: Training batch size (default: 32)
    
    Returns:
        Tuple of (result_a, result_b) BenchmarkResult objects
    
    Example:
        >>> neural_result, keras_result = compare_frameworks("neural", "keras")
        >>> reduction = (keras_result.lines_of_code - neural_result.lines_of_code) / keras_result.lines_of_code
        >>> print(f"Neural DSL uses {reduction*100:.1f}% fewer lines")
    """
    results = quick_benchmark(
        frameworks=[framework_a, framework_b],
        epochs=epochs,
        batch_size=batch_size,
    )
    
    result_a = next((r for r in results if r.framework.lower().replace(" ", "-") == framework_a), None)
    result_b = next((r for r in results if r.framework.lower().replace(" ", "-") == framework_b), None)
    
    return result_a, result_b


def get_available_frameworks():
    """
    Get list of available frameworks for benchmarking.
    
    Returns:
        List of framework names
    
    Example:
        >>> frameworks = get_available_frameworks()
        >>> print(f"Available: {', '.join(frameworks)}")
    """
    framework_classes = [
        ("Neural DSL", NeuralDSLImplementation),
        ("Keras", KerasImplementation),
        ("Raw TensorFlow", RawTensorFlowImplementation),
        ("PyTorch Lightning", PyTorchLightningImplementation),
        ("Raw PyTorch", RawPyTorchImplementation),
        ("Fast.ai", FastAIImplementation),
        ("Ludwig", LudwigImplementation),
    ]
    
    available = []
    for name, cls in framework_classes:
        try:
            cls()
            available.append(name)
        except ImportError:
            pass
    
    return available
