#!/usr/bin/env python
"""
Comprehensive demonstration of the Neural DSL benchmarking suite.

This example shows:
1. Running benchmarks programmatically
2. Customizing benchmark tasks
3. Collecting and analyzing metrics
4. Generating reports
5. Publishing results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural.benchmarks import (
    BenchmarkRunner,
    KerasImplementation,
    MetricsCollector,
    NeuralDSLImplementation,
    ReportGenerator,
)


def demonstrate_basic_benchmark():
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Benchmark")
    print("=" * 70)
    
    print("\nComparing Neural DSL vs Keras on MNIST classification...")
    
    frameworks = [
        NeuralDSLImplementation(),
        KerasImplementation(),
    ]
    
    runner = BenchmarkRunner(output_dir="demo_results", verbose=True)
    
    neural_result = runner.run_benchmark(
        framework_impl=frameworks[0],
        task_name="MNIST_Demo",
        dataset="mnist",
        epochs=2,
        batch_size=32,
        num_inference_samples=50,
    )
    
    keras_result = runner.run_benchmark(
        framework_impl=frameworks[1],
        task_name="MNIST_Demo",
        dataset="mnist",
        epochs=2,
        batch_size=32,
        num_inference_samples=50,
    )
    
    print("\n" + "-" * 70)
    print("RESULTS COMPARISON")
    print("-" * 70)
    
    print(f"\n{'Metric':<35} {'Neural DSL':<15} {'Keras':<15}")
    print("-" * 70)
    
    metrics = [
        ("Lines of Code", neural_result.lines_of_code, keras_result.lines_of_code),
        ("Training Time (s)", neural_result.training_time_seconds, keras_result.training_time_seconds),
        ("Inference Time (ms)", neural_result.inference_time_ms, keras_result.inference_time_ms),
        ("Accuracy", neural_result.model_accuracy, keras_result.model_accuracy),
        ("Setup Complexity", neural_result.setup_complexity, keras_result.setup_complexity),
        ("Readability Score", neural_result.code_readability_score, keras_result.code_readability_score),
    ]
    
    for name, neural_val, keras_val in metrics:
        print(f"{name:<35} {neural_val:<15.2f} {keras_val:<15.2f}")
    
    return [neural_result, keras_result]


def demonstrate_metrics_collection():
    print("\n" + "=" * 70)
    print("DEMO 2: Advanced Metrics Collection")
    print("=" * 70)
    
    collector = MetricsCollector()
    
    print("\nSystem Information:")
    system_info = collector.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    print("\nCollecting performance metrics...")
    collector.start_collection()
    
    import time
    import numpy as np
    
    for i in range(10):
        large_array = np.random.rand(1000, 1000)
        _ = np.dot(large_array, large_array.T)
        collector.collect_snapshot()
        time.sleep(0.1)
        del large_array
    
    summary = collector.get_summary()
    
    print("\nPerformance Summary:")
    print(f"  Peak Memory: {summary['peak_memory_mb']:.2f} MB")
    print(f"  Average Memory: {summary['avg_memory_mb']:.2f} MB")
    print(f"  Average CPU: {summary['avg_cpu_percent']:.2f}%")
    print(f"  Total Time: {summary['total_time_seconds']:.2f}s")


def demonstrate_batch_processing():
    print("\n" + "=" * 70)
    print("DEMO 3: Batch Processing Multiple Tasks")
    print("=" * 70)
    
    frameworks = [
        NeuralDSLImplementation(),
        KerasImplementation(),
    ]
    
    tasks = [
        {
            "name": "MNIST_Fast",
            "dataset": "mnist",
            "epochs": 1,
            "batch_size": 64,
        },
        {
            "name": "MNIST_Accurate",
            "dataset": "mnist",
            "epochs": 3,
            "batch_size": 32,
        },
    ]
    
    print(f"\nRunning {len(frameworks)} frameworks × {len(tasks)} tasks...")
    
    runner = BenchmarkRunner(output_dir="demo_batch_results", verbose=False)
    results = runner.run_all_benchmarks(frameworks, tasks, save_results=True)
    
    print(f"\n✓ Completed {len(results)} benchmarks")
    
    comparison = runner.compare_frameworks()
    
    print("\nFramework Averages:")
    for framework, metrics in comparison.items():
        print(f"\n{framework}:")
        print(f"  Avg LOC: {metrics['avg_lines_of_code']:.0f}")
        print(f"  Avg Training Time: {metrics['avg_training_time']:.2f}s")
        print(f"  Avg Accuracy: {metrics['avg_accuracy']:.4f}")
    
    return results


def demonstrate_report_generation(results):
    print("\n" + "=" * 70)
    print("DEMO 4: Report Generation")
    print("=" * 70)
    
    report_gen = ReportGenerator(output_dir="demo_reports")
    
    print("\nGenerating comprehensive report...")
    report_path = report_gen.generate_report(
        [r.to_dict() for r in results],
        report_name="demo_benchmark",
        include_plots=True,
    )
    
    print(f"\n✓ Report generated: {report_path}")
    print(f"\nThe report includes:")
    print("  - Interactive HTML dashboard")
    print("  - Markdown summary")
    print("  - Raw JSON data")
    print("  - Reproducibility script")
    print("  - Comparison charts (PNG)")
    
    report_dir = Path(report_path).parent
    print(f"\nGenerated files:")
    for file in sorted(report_dir.glob("*")):
        print(f"  - {file.name}")
    
    return report_path


def demonstrate_custom_implementation():
    print("\n" + "=" * 70)
    print("DEMO 5: Custom Framework Implementation")
    print("=" * 70)
    
    print("\nExample: Creating a custom framework adapter")
    
    code_example = '''
from neural.benchmarks.framework_implementations import FrameworkImplementation

class CustomFrameworkImpl(FrameworkImplementation):
    def __init__(self):
        super().__init__("MyCustomFramework")
    
    def setup(self):
        self.code_content = """
import myframework as mf

model = mf.Sequential([
    mf.layers.Dense(128),
    mf.layers.Dense(10),
])
"""
    
    def build_model(self):
        # Build actual model
        self.model = create_model()
    
    def train(self, dataset, epochs, batch_size):
        # Train and return metrics
        return {
            "training_time": ...,
            "accuracy": ...,
        }
    
    def predict_single(self):
        return self.model.predict(sample)
    
    def _save_model(self, path):
        self.model.save(path)
    
    def get_parameter_count(self):
        return count_params(self.model)
'''
    
    print(code_example)
    print("\nThen use it in benchmarks:")
    print("  frameworks = [CustomFrameworkImpl(), ...]")
    print("  runner.run_all_benchmarks(frameworks, tasks)")


def main():
    print("\n" + "=" * 70)
    print("Neural DSL Benchmarking Suite - Comprehensive Demo")
    print("=" * 70)
    
    try:
        # Demo 1: Basic benchmark
        basic_results = demonstrate_basic_benchmark()
        
        # Demo 2: Metrics collection
        demonstrate_metrics_collection()
        
        # Demo 3: Batch processing
        batch_results = demonstrate_batch_processing()
        
        # Demo 4: Report generation
        all_results = basic_results + batch_results
        report_path = demonstrate_report_generation(all_results)
        
        # Demo 5: Custom implementation
        demonstrate_custom_implementation()
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETE!")
        print("=" * 70)
        
        print("\nWhat we demonstrated:")
        print("  ✓ Basic benchmark comparison")
        print("  ✓ Advanced metrics collection")
        print("  ✓ Batch processing")
        print("  ✓ Report generation")
        print("  ✓ Custom framework implementation")
        
        print(f"\nView the report at:")
        print(f"  file://{Path(report_path).absolute()}")
        
        print("\nNext steps:")
        print("  1. Run full benchmarks: python neural/benchmarks/run_benchmarks.py")
        print("  2. Read documentation: docs/BENCHMARKS.md")
        print("  3. Add custom frameworks: see neural/benchmarks/CONTRIBUTING.md")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
