#!/usr/bin/env python
"""
Example script demonstrating basic benchmark usage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.benchmarks import (
    BenchmarkRunner,
    KerasImplementation,
    NeuralDSLImplementation,
    ReportGenerator,
)


def run_simple_benchmark():
    print("Running simple benchmark example...")
    print("=" * 60)
    
    frameworks = [
        NeuralDSLImplementation(),
        KerasImplementation(),
    ]
    
    runner = BenchmarkRunner(output_dir="example_results", verbose=True)
    
    result_neural = runner.run_benchmark(
        framework_impl=frameworks[0],
        task_name="MNIST_Example",
        dataset="mnist",
        epochs=3,
        batch_size=32,
        num_inference_samples=50,
    )
    
    result_keras = runner.run_benchmark(
        framework_impl=frameworks[1],
        task_name="MNIST_Example",
        dataset="mnist",
        epochs=3,
        batch_size=32,
        num_inference_samples=50,
    )
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<30} {'Neural DSL':<15} {'Keras':<15} {'Winner'}")
    print("-" * 75)
    
    metrics = [
        ("Lines of Code", result_neural.lines_of_code, result_keras.lines_of_code, "lower"),
        ("Training Time (s)", result_neural.training_time_seconds, result_keras.training_time_seconds, "lower"),
        ("Inference Time (ms)", result_neural.inference_time_ms, result_keras.inference_time_ms, "lower"),
        ("Accuracy", result_neural.model_accuracy, result_keras.model_accuracy, "higher"),
        ("Model Size (MB)", result_neural.model_size_mb, result_keras.model_size_mb, "lower"),
        ("Setup Complexity", result_neural.setup_complexity, result_keras.setup_complexity, "lower"),
    ]
    
    for name, neural_val, keras_val, direction in metrics:
        if direction == "lower":
            winner = "Neural DSL" if neural_val < keras_val else "Keras"
        else:
            winner = "Neural DSL" if neural_val > keras_val else "Keras"
        
        print(f"{name:<30} {neural_val:<15.2f} {keras_val:<15.2f} {winner}")
    
    print("\n" + "=" * 60)
    print("Generating report...")
    
    report_gen = ReportGenerator(output_dir="example_reports")
    report_path = report_gen.generate_report(
        [result_neural.to_dict(), result_keras.to_dict()],
        report_name="example_benchmark",
        include_plots=True,
    )
    
    print(f"✓ Report generated: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_simple_benchmark()
