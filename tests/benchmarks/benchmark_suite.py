"""
Comprehensive benchmark suite orchestrator.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from .benchmark_runner import BenchmarkRunner
from .models import get_benchmark_models


class BenchmarkSuite:
    """Orchestrates comprehensive benchmarking across models and backends."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.runner = BenchmarkRunner(output_dir)
        self.models = get_benchmark_models()
        self.results = []
        
    def run_all_benchmarks(
        self,
        backends: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        epochs: int = 5,
        dataset: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Run all benchmarks for specified backends and models."""
        if backends is None:
            backends = ['tensorflow', 'pytorch']
        
        if models is None:
            models = list(self.models.keys())
        
        if dataset is None:
            dataset = self._load_mnist()
        
        all_results = []
        
        for backend in backends:
            for model_name in models:
                if model_name not in self.models:
                    print(f"Warning: Model {model_name} not found, skipping")
                    continue
                
                print(f"\nBenchmarking {model_name} on {backend}...")
                
                model_def = self.models[model_name]
                
                print("  Running Neural DSL version...")
                neural_results = self.runner.benchmark_neural_dsl(
                    model_name=model_name,
                    dsl_code=model_def['neural_dsl'],
                    backend=backend,
                    dataset=dataset,
                    epochs=epochs
                )
                
                print(f"  Running native {backend} version...")
                native_results = self.runner.benchmark_native(
                    model_name=model_name,
                    native_code=model_def[backend],
                    backend=backend,
                    dataset=dataset,
                    epochs=epochs
                )
                
                comparison = self.runner.compare_results(neural_results, native_results)
                
                all_results.append({
                    'neural_dsl': neural_results,
                    'native': native_results,
                    'comparison': comparison
                })
                
                self._print_comparison(comparison)
        
        self.results = all_results
        return all_results
    
    def _load_mnist(self):
        """Load MNIST dataset."""
        try:
            import tensorflow as tf
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            return (x_train, y_train), (x_test, y_test)
        except Exception as e:
            print(f"Warning: Failed to load MNIST: {e}")
            print("Using synthetic data for benchmarking")
            import numpy as np
            x_train = np.random.rand(1000, 28, 28).astype('float32')
            y_train = np.random.randint(0, 10, 1000)
            x_test = np.random.rand(200, 28, 28).astype('float32')
            y_test = np.random.randint(0, 10, 200)
            return (x_train, y_train), (x_test, y_test)
    
    def _print_comparison(self, comparison: Dict[str, Any]):
        """Print comparison results in readable format."""
        print(f"\n  Results for {comparison['model_name']} ({comparison['backend']}):")
        print("    Overhead:")
        print(f"      Parse time: {comparison['overhead']['parse_time']:.4f}s")
        print(f"      Codegen time: {comparison['overhead']['codegen_time']:.4f}s")
        print(f"      Total overhead: {comparison['overhead']['total_overhead']:.4f}s")
        
        print("    Training time:")
        print(f"      Neural DSL: {comparison['training_time']['neural_dsl']:.4f}s")
        print(f"      Native: {comparison['training_time']['native']:.4f}s")
        print(f"      Difference: {comparison['training_time']['difference']:.4f}s ({comparison['training_time']['percentage']:.2f}%)")
        
        print("    Memory usage:")
        print(f"      Neural DSL: {comparison['memory']['neural_dsl_mb']:.2f} MB")
        print(f"      Native: {comparison['memory']['native_mb']:.2f} MB")
        print(f"      Difference: {comparison['memory']['difference_mb']:.2f} MB ({comparison['memory']['percentage']:.2f}%)")
        
        print("    Model performance:")
        print(f"      Neural DSL accuracy: {comparison['accuracy']['neural_dsl']:.4f}")
        print(f"      Native accuracy: {comparison['accuracy']['native']:.4f}")
        print(f"      Accuracy difference: {comparison['accuracy']['difference']:.4f}")
        print(f"      Neural DSL loss: {comparison['loss']['neural_dsl']:.4f}")
        print(f"      Native loss: {comparison['loss']['native']:.4f}")
    
    def generate_markdown_report(self, output_file: str = 'benchmark_report.md') -> str:
        """Generate markdown report from benchmark results."""
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmarks first.")
        
        report_lines = [
            "# Neural DSL Benchmark Results\n",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n## Summary\n",
            f"Total benchmarks run: {len(self.results)}\n",
            "\n## Detailed Results\n"
        ]
        
        for result in self.results:
            comp = result['comparison']
            report_lines.extend([
                f"\n### {comp['model_name']} - {comp['backend']}\n",
                "\n#### Compilation Overhead\n",
                "| Metric | Time (s) |",
                "|--------|----------|",
                f"| Parse time | {comp['overhead']['parse_time']:.4f} |",
                f"| Code generation | {comp['overhead']['codegen_time']:.4f} |",
                f"| **Total overhead** | **{comp['overhead']['total_overhead']:.4f}** |",
                
                "\n#### Training Performance\n",
                "| Implementation | Time (s) | vs Native |",
                "|----------------|----------|-----------|",
                f"| Neural DSL | {comp['training_time']['neural_dsl']:.4f} | {comp['training_time']['percentage']:+.2f}% |",
                f"| Native {comp['backend'].title()} | {comp['training_time']['native']:.4f} | 0.00% |",
                
                "\n#### Memory Usage\n",
                "| Implementation | Memory (MB) | vs Native |",
                "|----------------|-------------|-----------|",
                f"| Neural DSL | {comp['memory']['neural_dsl_mb']:.2f} | {comp['memory']['percentage']:+.2f}% |",
                f"| Native {comp['backend'].title()} | {comp['memory']['native_mb']:.2f} | 0.00% |",
                
                "\n#### Model Performance\n",
                "| Metric | Neural DSL | Native | Difference |",
                "|--------|-----------|--------|------------|",
                f"| Final Accuracy | {comp['accuracy']['neural_dsl']:.4f} | {comp['accuracy']['native']:.4f} | {comp['accuracy']['difference']:+.4f} |",
                f"| Final Loss | {comp['loss']['neural_dsl']:.4f} | {comp['loss']['native']:.4f} | {comp['loss']['difference']:+.4f} |",
                ""
            ])
        
        report_lines.extend([
            "\n## Aggregate Statistics\n",
            self._generate_aggregate_stats()
        ])
        
        report_content = '\n'.join(report_lines)
        
        output_path = Path(self.runner.output_dir) / output_file
        output_path.write_text(report_content)
        
        return str(output_path)
    
    def _generate_aggregate_stats(self) -> str:
        """Generate aggregate statistics across all benchmarks."""
        if not self.results:
            return "No results available."
        
        avg_overhead = sum(r['comparison']['overhead']['total_overhead'] for r in self.results) / len(self.results)
        avg_train_diff = sum(r['comparison']['training_time']['difference'] for r in self.results) / len(self.results)
        avg_train_pct = sum(r['comparison']['training_time']['percentage'] for r in self.results) / len(self.results)
        avg_mem_diff = sum(r['comparison']['memory']['difference_mb'] for r in self.results) / len(self.results)
        avg_mem_pct = sum(r['comparison']['memory']['percentage'] for r in self.results) / len(self.results)
        avg_acc_diff = sum(r['comparison']['accuracy']['difference'] for r in self.results) / len(self.results)
        
        lines = [
            "| Metric | Average Value |",
            "|--------|---------------|",
            f"| Compilation overhead | {avg_overhead:.4f}s |",
            f"| Training time difference | {avg_train_diff:+.4f}s ({avg_train_pct:+.2f}%) |",
            f"| Memory usage difference | {avg_mem_diff:+.2f} MB ({avg_mem_pct:+.2f}%) |",
            f"| Accuracy difference | {avg_acc_diff:+.4f} |",
            "",
            "### Key Findings\n",
            f"- **Compilation overhead**: Neural DSL adds an average of {avg_overhead:.4f}s for parsing and code generation.",
            f"- **Training performance**: Neural DSL is {avg_train_pct:+.2f}% compared to native implementations.",
            f"- **Memory efficiency**: Neural DSL uses {avg_mem_pct:+.2f}% more/less memory on average.",
            f"- **Model quality**: Neural DSL achieves comparable accuracy (±{abs(avg_acc_diff):.4f} on average).",
        ]
        
        return '\n'.join(lines)
    
    def save_results_json(self, filename: str = 'benchmark_results.json') -> str:
        """Save all results to JSON file."""
        return self.runner.save_results(self.results, filename)
