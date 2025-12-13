"""
Core benchmark runner for comparing Neural DSL against other frameworks.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkResult:
    framework: str
    task_name: str
    lines_of_code: int
    development_time_seconds: float
    training_time_seconds: float
    inference_time_ms: float
    peak_memory_mb: float
    model_accuracy: float
    model_size_mb: float
    parameters_count: int
    compilation_time_seconds: float
    setup_complexity: int
    code_readability_score: float
    error_rate: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "task_name": self.task_name,
            "lines_of_code": self.lines_of_code,
            "development_time_seconds": self.development_time_seconds,
            "training_time_seconds": self.training_time_seconds,
            "inference_time_ms": self.inference_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "model_accuracy": self.model_accuracy,
            "model_size_mb": self.model_size_mb,
            "parameters_count": self.parameters_count,
            "compilation_time_seconds": self.compilation_time_seconds,
            "setup_complexity": self.setup_complexity,
            "code_readability_score": self.code_readability_score,
            "error_rate": self.error_rate,
            "additional_metrics": self.additional_metrics,
            "timestamp": self.timestamp,
        }


class BenchmarkRunner:
    def __init__(self, output_dir: str = "benchmark_results", verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        framework_impl,
        task_name: str,
        dataset: str,
        epochs: int = 5,
        batch_size: int = 32,
        num_inference_samples: int = 100,
    ) -> BenchmarkResult:
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running benchmark: {framework_impl.name} on {task_name}")
            print(f"{'='*60}")

        result = None
        try:
            start_time = time.time()
            framework_impl.setup()
            setup_time = time.time() - start_time

            start_time = time.time()
            framework_impl.build_model()
            build_time = time.time() - start_time

            metrics = framework_impl.train(
                dataset=dataset, epochs=epochs, batch_size=batch_size
            )

            inference_times = []
            for _ in range(num_inference_samples):
                start = time.time()
                framework_impl.predict_single()
                inference_times.append((time.time() - start) * 1000)

            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)

            loc = framework_impl.count_lines_of_code()
            code_complexity = framework_impl.get_code_complexity()
            model_size = framework_impl.get_model_size_mb()
            param_count = framework_impl.get_parameter_count()

            result = BenchmarkResult(
                framework=framework_impl.name,
                task_name=task_name,
                lines_of_code=loc,
                development_time_seconds=setup_time + build_time,
                training_time_seconds=metrics.get("training_time", 0),
                inference_time_ms=avg_inference_time,
                peak_memory_mb=metrics.get("peak_memory_mb", 0),
                model_accuracy=metrics.get("accuracy", 0),
                model_size_mb=model_size,
                parameters_count=param_count,
                compilation_time_seconds=build_time,
                setup_complexity=code_complexity["setup_complexity"],
                code_readability_score=code_complexity["readability_score"],
                error_rate=metrics.get("error_rate", 0),
                additional_metrics={
                    "inference_time_std_ms": std_inference_time,
                    "throughput_samples_per_sec": 1000 / avg_inference_time if avg_inference_time > 0 else 0,
                    "val_accuracy": metrics.get("val_accuracy", 0),
                    "val_loss": metrics.get("val_loss", 0),
                    "training_loss": metrics.get("training_loss", 0),
                },
            )

            self.results.append(result)

            if self.verbose:
                print(f"\n✓ Benchmark completed for {framework_impl.name}")
                print(f"  Lines of Code: {loc}")
                print(f"  Training Time: {metrics.get('training_time', 0):.2f}s")
                print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"  Inference Time: {avg_inference_time:.2f}ms")

        except Exception as e:
            if self.verbose:
                print(f"\n✗ Benchmark failed for {framework_impl.name}: {str(e)}")
            raise

        finally:
            framework_impl.cleanup()

        return result

    def run_all_benchmarks(
        self,
        frameworks: List[Any],
        tasks: List[Dict[str, Any]],
        save_results: bool = True,
    ) -> List[BenchmarkResult]:
        all_results = []

        for task in tasks:
            task_name = task["name"]
            dataset = task["dataset"]
            epochs = task.get("epochs", 5)
            batch_size = task.get("batch_size", 32)

            for framework_impl in frameworks:
                try:
                    result = self.run_benchmark(
                        framework_impl=framework_impl,
                        task_name=task_name,
                        dataset=dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                    )
                    all_results.append(result)
                except Exception as e:
                    if self.verbose:
                        print(f"Skipping {framework_impl.name} for {task_name}: {e}")

        if save_results:
            self.save_results(all_results)

        return all_results

    def save_results(self, results: Optional[List[BenchmarkResult]] = None):
        if results is None:
            results = self.results

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_results_{timestamp}.json"

        results_dict = [r.to_dict() for r in results]

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        if self.verbose:
            print(f"\n✓ Results saved to {output_file}")

        return output_file

    def load_results(self, filepath: str) -> List[BenchmarkResult]:
        with open(filepath, "r") as f:
            data = json.load(f)

        results = []
        for item in data:
            result = BenchmarkResult(**item)
            results.append(result)

        return results

    def compare_frameworks(
        self, results: Optional[List[BenchmarkResult]] = None
    ) -> Dict[str, Any]:
        if results is None:
            results = self.results

        if not results:
            return {}

        frameworks = {}
        for result in results:
            if result.framework not in frameworks:
                frameworks[result.framework] = []
            frameworks[result.framework].append(result)

        comparison = {}
        for framework, fw_results in frameworks.items():
            comparison[framework] = {
                "avg_lines_of_code": np.mean([r.lines_of_code for r in fw_results]),
                "avg_training_time": np.mean([r.training_time_seconds for r in fw_results]),
                "avg_inference_time": np.mean([r.inference_time_ms for r in fw_results]),
                "avg_accuracy": np.mean([r.model_accuracy for r in fw_results]),
                "avg_model_size": np.mean([r.model_size_mb for r in fw_results]),
                "avg_setup_complexity": np.mean([r.setup_complexity for r in fw_results]),
                "avg_readability": np.mean([r.code_readability_score for r in fw_results]),
            }

        return comparison
