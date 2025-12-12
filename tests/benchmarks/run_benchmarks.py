#!/usr/bin/env python
"""
Standalone script to run comprehensive benchmarks.

Usage:
    python tests/benchmarks/run_benchmarks.py [options]

Options:
    --backends BACKENDS     Comma-separated list of backends (tensorflow,pytorch)
    --models MODELS         Comma-separated list of models (simple_mlp,cnn,deep_mlp)
    --epochs EPOCHS         Number of training epochs (default: 5)
    --output OUTPUT         Output directory for results (default: ./benchmark_results)
    --report                Generate markdown report
    --json                  Save results as JSON
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.benchmark_suite import BenchmarkSuite


def main():
    parser = argparse.ArgumentParser(
        description='Run Neural DSL benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--backends',
        type=str,
        default='tensorflow,pytorch',
        help='Comma-separated list of backends (default: tensorflow,pytorch)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='simple_mlp,cnn,deep_mlp',
        help='Comma-separated list of models (default: simple_mlp,cnn,deep_mlp)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./benchmark_results',
        help='Output directory for results (default: ./benchmark_results)'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate markdown report'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Save results as JSON'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick benchmark with fewer epochs (2 instead of 5)'
    )
    
    args = parser.parse_args()
    
    backends = [b.strip() for b in args.backends.split(',')]
    models = [m.strip() for m in args.models.split(',')]
    epochs = 2 if args.quick else args.epochs
    
    print("=" * 80)
    print("Neural DSL Comprehensive Benchmark Suite")
    print("=" * 80)
    print(f"Backends: {', '.join(backends)}")
    print(f"Models: {', '.join(models)}")
    print(f"Epochs: {epochs}")
    print(f"Output directory: {args.output}")
    print("=" * 80)
    print()
    
    suite = BenchmarkSuite(output_dir=args.output)
    
    print("Running benchmarks...")
    results = suite.run_all_benchmarks(
        backends=backends,
        models=models,
        epochs=epochs
    )
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
    
    if args.report:
        print("\nGenerating markdown report...")
        report_path = suite.generate_markdown_report('benchmark_report.md')
        print(f"Report saved to: {report_path}")
    
    if args.json:
        print("\nSaving JSON results...")
        json_path = suite.save_results_json('benchmark_results.json')
        print(f"JSON saved to: {json_path}")
    
    print(f"\nAll results saved to: {args.output}")
    print("\nTo copy results to docs directory:")
    print(f"  cp {args.output}/benchmark_report.md docs/benchmarks.md")


if __name__ == '__main__':
    main()
