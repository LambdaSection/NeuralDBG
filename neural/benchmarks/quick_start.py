#!/usr/bin/env python
"""
Quick start script for Neural DSL benchmarking.

This script demonstrates the simplest way to run benchmarks and generate reports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    print("=" * 70)
    print("Neural DSL Benchmarking Suite - Quick Start")
    print("=" * 70)
    print()
    
    print("This script will:")
    print("  1. Benchmark Neural DSL vs Keras")
    print("  2. Train on MNIST for 3 epochs")
    print("  3. Generate an HTML report with visualizations")
    print()
    
    input("Press Enter to continue...")
    print()
    
    try:
        from neural.benchmarks import (
            BenchmarkRunner,
            KerasImplementation,
            NeuralDSLImplementation,
            ReportGenerator,
        )
        
        print("✓ Imported benchmark modules")
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        print("\nPlease install Neural DSL:")
        print("  pip install -e '.[full]'")
        return 1
    
    frameworks = [
        NeuralDSLImplementation(),
        KerasImplementation(),
    ]
    
    print(f"✓ Loaded {len(frameworks)} frameworks")
    print()
    
    tasks = [{
        "name": "MNIST_QuickStart",
        "dataset": "mnist",
        "epochs": 3,
        "batch_size": 32,
    }]
    
    print("Starting benchmarks...")
    print("-" * 70)
    
    try:
        runner = BenchmarkRunner(output_dir="quick_start_results", verbose=True)
        results = runner.run_all_benchmarks(frameworks, tasks, save_results=True)
        
        print("-" * 70)
        print(f"✓ Completed {len(results)} benchmark(s)")
        print()
        
        print("Generating report...")
        report_gen = ReportGenerator(output_dir="quick_start_reports")
        report_path = report_gen.generate_report(
            [r.to_dict() for r in results],
            report_name="quick_start",
            include_plots=True,
        )
        
        print()
        print("=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"\nReport available at: {report_path}")
        print(f"\nTo view the report:")
        print(f"  1. Open your web browser")
        print(f"  2. Navigate to: file://{Path(report_path).absolute()}")
        print()
        print("Next steps:")
        print("  - Run full benchmarks: python neural/benchmarks/run_benchmarks.py")
        print("  - Read documentation: docs/BENCHMARKS.md")
        print("  - View examples: neural/benchmarks/example_benchmark.py")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"\n✗ Benchmark failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("If this is a dependency issue, try:")
        print("  pip install tensorflow")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
