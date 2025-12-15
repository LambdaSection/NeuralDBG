#!/usr/bin/env python
"""
Example benchmark demonstrating Neural DSL performance advantages.

This script showcases:
1. Setting up multiple framework implementations
2. Running fair comparisons
3. Generating comprehensive reports
4. Publishing results to website

Usage:
    python neural/benchmarks/example_benchmark.py
    python neural/benchmarks/example_benchmark.py --quick
    python neural/benchmarks/example_benchmark.py --frameworks neural keras pytorch-lightning
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.benchmarks.benchmark_runner import BenchmarkRunner
from neural.benchmarks.framework_implementations import (
    KerasImplementation,
    NeuralDSLImplementation,
    PyTorchLightningImplementation,
    RawPyTorchImplementation,
    RawTensorFlowImplementation,
)
from neural.benchmarks.report_generator import ReportGenerator


def quick_benchmark():
    """
    Run a quick benchmark with minimal frameworks.
    Good for testing and development.
    """
    print("=" * 70)
    print("QUICK BENCHMARK MODE")
    print("=" * 70)
    print("\nComparing Neural DSL vs. Keras (fast comparison)\n")
    
    frameworks = [
        NeuralDSLImplementation(),
        KerasImplementation(),
    ]
    
    tasks = [
        {
            "name": "MNIST_Quick_Test",
            "dataset": "mnist",
            "epochs": 2,
            "batch_size": 64,
        }
    ]
    
    runner = BenchmarkRunner(output_dir="benchmark_results", verbose=True)
    results = runner.run_all_benchmarks(frameworks, tasks)
    
    return results


def comprehensive_benchmark():
    """
    Run comprehensive benchmarks across all available frameworks.
    This is the full benchmark suite used for marketing materials.
    """
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK MODE")
    print("=" * 70)
    print("\nRunning full benchmark suite...\n")
    
    frameworks = []
    
    print("Loading frameworks:")
    
    try:
        frameworks.append(NeuralDSLImplementation())
        print("  ✓ Neural DSL")
    except Exception as e:
        print(f"  ✗ Neural DSL: {e}")
    
    try:
        frameworks.append(KerasImplementation())
        print("  ✓ Keras")
    except Exception as e:
        print(f"  ✗ Keras: {e}")
    
    try:
        frameworks.append(RawTensorFlowImplementation())
        print("  ✓ Raw TensorFlow")
    except Exception as e:
        print(f"  ✗ Raw TensorFlow: {e}")
    
    try:
        frameworks.append(PyTorchLightningImplementation())
        print("  ✓ PyTorch Lightning")
    except Exception as e:
        print(f"  ✗ PyTorch Lightning: {e}")
    
    try:
        frameworks.append(RawPyTorchImplementation())
        print("  ✓ Raw PyTorch")
    except Exception as e:
        print(f"  ✗ Raw PyTorch: {e}")
    
    print()
    
    if not frameworks:
        print("✗ No frameworks available to benchmark!")
        sys.exit(1)
    
    tasks = [
        {
            "name": "MNIST_Image_Classification",
            "dataset": "mnist",
            "epochs": 5,
            "batch_size": 32,
        }
    ]
    
    runner = BenchmarkRunner(output_dir="benchmark_results", verbose=True)
    results = runner.run_all_benchmarks(frameworks, tasks)
    
    return results


def custom_benchmark(framework_list, epochs=5, batch_size=32):
    """
    Run a custom benchmark with specified frameworks and parameters.
    
    Args:
        framework_list: List of framework names to benchmark
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    print("=" * 70)
    print("CUSTOM BENCHMARK MODE")
    print("=" * 70)
    print(f"\nFrameworks: {', '.join(framework_list)}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}\n")
    
    framework_map = {
        "neural": NeuralDSLImplementation,
        "keras": KerasImplementation,
        "raw-tensorflow": RawTensorFlowImplementation,
        "pytorch-lightning": PyTorchLightningImplementation,
        "raw-pytorch": RawPyTorchImplementation,
    }
    
    frameworks = []
    for fw_name in framework_list:
        if fw_name in framework_map:
            try:
                frameworks.append(framework_map[fw_name]())
                print(f"  ✓ Loaded {fw_name}")
            except Exception as e:
                print(f"  ✗ Failed to load {fw_name}: {e}")
    
    if not frameworks:
        print("\n✗ No valid frameworks selected!")
        sys.exit(1)
    
    tasks = [
        {
            "name": f"MNIST_Custom_{epochs}ep_{batch_size}bs",
            "dataset": "mnist",
            "epochs": epochs,
            "batch_size": batch_size,
        }
    ]
    
    runner = BenchmarkRunner(output_dir="benchmark_results", verbose=True)
    results = runner.run_all_benchmarks(frameworks, tasks)
    
    return results


def print_results_summary(results):
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    if not results:
        print("\nNo results to display.")
        return
    
    print(f"\nTotal benchmarks: {len(results)}")
    print(f"Frameworks tested: {len(set(r.framework for r in results))}")
    
    print("\n" + "-" * 70)
    print(f"{'Framework':<25} {'LOC':<8} {'Accuracy':<10} {'Train(s)':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result.framework:<25} "
              f"{result.lines_of_code:<8} "
              f"{result.model_accuracy:<10.4f} "
              f"{result.training_time_seconds:<10.2f}")
    
    print("-" * 70)
    
    neural_results = [r for r in results if r.framework == "Neural DSL"]
    if neural_results:
        other_results = [r for r in results if r.framework != "Neural DSL"]
        if other_results:
            neural_loc = sum(r.lines_of_code for r in neural_results) / len(neural_results)
            other_loc = sum(r.lines_of_code for r in other_results) / len(other_results)
            reduction = ((other_loc - neural_loc) / other_loc) * 100
            
            print(f"\n✓ Neural DSL code reduction: {reduction:.1f}%")
            print(f"✓ Neural DSL avg LOC: {neural_loc:.0f}")
            print(f"✓ Other frameworks avg LOC: {other_loc:.0f}")


def generate_report(results, include_plots=True):
    """Generate HTML and markdown reports."""
    print("\n" + "=" * 70)
    print("GENERATING REPORTS")
    print("=" * 70)
    
    report_gen = ReportGenerator(output_dir="benchmark_reports")
    report_path = report_gen.generate_report(
        [r.to_dict() for r in results],
        report_name="neural_dsl_benchmark",
        include_plots=include_plots,
    )
    
    print(f"\n✓ Reports generated successfully!")
    print(f"\nView results:")
    print(f"  - HTML: file://{Path(report_path).absolute()}")
    print(f"  - Markdown: {Path(report_path).parent / 'README.md'}")


def main():
    """Main entry point for example benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Neural DSL benchmark example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (2 frameworks, 2 epochs)
  python example_benchmark.py --quick
  
  # Comprehensive (all frameworks, 5 epochs)
  python example_benchmark.py --comprehensive
  
  # Custom frameworks
  python example_benchmark.py --frameworks neural keras raw-pytorch
  
  # Custom parameters
  python example_benchmark.py --frameworks neural keras --epochs 10 --batch-size 64
  
  # Skip plots for faster generation
  python example_benchmark.py --no-plots
        """
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (Neural DSL vs Keras only)",
    )
    
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive benchmark (all frameworks)",
    )
    
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["neural", "keras", "raw-tensorflow", "pytorch-lightning", "raw-pytorch"],
        help="Specific frameworks to benchmark",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (faster)",
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating reports",
    )
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            results = quick_benchmark()
        elif args.comprehensive or not args.frameworks:
            results = comprehensive_benchmark()
        else:
            results = custom_benchmark(
                args.frameworks,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        
        print_results_summary(results)
        
        if not args.no_report:
            generate_report(results, include_plots=not args.no_plots)
        
        print("\n" + "=" * 70)
        print("✓ BENCHMARK COMPLETE")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
