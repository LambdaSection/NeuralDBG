#!/usr/bin/env python
"""
Main script to run comprehensive Neural DSL benchmarks.

This script compares Neural DSL against Keras, PyTorch Lightning, Fast.ai, and Ludwig
across multiple metrics including development time, lines of code, training speed,
and model performance.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.benchmarks.benchmark_runner import BenchmarkRunner
from neural.benchmarks.framework_implementations import (
    FastAIImplementation,
    KerasImplementation,
    LudwigImplementation,
    NeuralDSLImplementation,
    PyTorchLightningImplementation,
    RawPyTorchImplementation,
    RawTensorFlowImplementation,
)
from neural.benchmarks.report_generator import ReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Neural DSL benchmarks against other frameworks"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["neural", "keras", "raw-tensorflow", "pytorch-lightning", "raw-pytorch", "fastai", "ludwig", "all"],
        default=["all"],
        help="Frameworks to benchmark (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--report-dir",
        default="benchmark_reports",
        help="Output directory for reports (default: benchmark_reports)",
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
        "--skip-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.quiet:
        print("=" * 70)
        print("Neural DSL Comprehensive Benchmarking Suite")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Frameworks: {', '.join(args.frameworks)}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Output Directory: {args.output_dir}")
        print(f"  Report Directory: {args.report_dir}")
        print()
    
    framework_map = {
        "neural": NeuralDSLImplementation,
        "keras": KerasImplementation,
        "pytorch-lightning": PyTorchLightningImplementation,
        "fastai": FastAIImplementation,
        "ludwig": LudwigImplementation,
    }
    
    if "all" in args.frameworks:
        selected_frameworks = list(framework_map.keys())
    else:
        selected_frameworks = args.frameworks
    
    frameworks = []
    for fw_name in selected_frameworks:
        try:
            fw_class = framework_map[fw_name]
            frameworks.append(fw_class())
            if not args.quiet:
                print(f"✓ Loaded {fw_name}")
        except ImportError as e:
            if not args.quiet:
                print(f"⚠ Skipping {fw_name}: {e}")
        except Exception as e:
            if not args.quiet:
                print(f"✗ Error loading {fw_name}: {e}")
    
    if not frameworks:
        print("\n✗ No frameworks available to benchmark!")
        sys.exit(1)
    
    tasks = [
        {
            "name": "MNIST_Image_Classification",
            "dataset": "mnist",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
    ]
    
    runner = BenchmarkRunner(output_dir=args.output_dir, verbose=not args.quiet)
    
    if not args.quiet:
        print(f"\n{'=' * 70}")
        print(f"Running benchmarks: {len(frameworks)} framework(s) × {len(tasks)} task(s)")
        print(f"{'=' * 70}\n")
    
    try:
        results = runner.run_all_benchmarks(frameworks, tasks, save_results=True)
        
        if not args.quiet:
            print(f"\n{'=' * 70}")
            print("Benchmark Execution Complete")
            print(f"{'=' * 70}\n")
        
        report_gen = ReportGenerator(output_dir=args.report_dir)
        report_path = report_gen.generate_report(
            [r.to_dict() for r in results],
            report_name="neural_dsl_benchmark",
            include_plots=not args.skip_plots,
        )
        
        if not args.quiet:
            print(f"\n{'=' * 70}")
            print("Benchmarking Complete!")
            print(f"{'=' * 70}")
            print(f"\n✓ Results saved to: {args.output_dir}")
            print(f"✓ Report available at: {report_path}")
            print(f"\nTo view the report, open: file://{Path(report_path).absolute()}")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmarking interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
xc()
        sys.exit(1)


if __name__ == "__main__":
    main()
