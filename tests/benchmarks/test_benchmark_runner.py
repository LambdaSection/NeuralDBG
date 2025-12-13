"""
Tests for the benchmark runner.
"""

import json
import tempfile
from pathlib import Path

import pytest

from neural.benchmarks.benchmark_runner import BenchmarkResult, BenchmarkRunner
from neural.benchmarks.framework_implementations import KerasImplementation, NeuralDSLImplementation


class TestBenchmarkResult:
    def test_benchmark_result_creation(self):
        result = BenchmarkResult(
            framework="Test Framework",
            task_name="Test Task",
            lines_of_code=10,
            development_time_seconds=5.0,
            training_time_seconds=30.0,
            inference_time_ms=2.5,
            peak_memory_mb=100.0,
            model_accuracy=0.95,
            model_size_mb=5.0,
            parameters_count=1000,
            compilation_time_seconds=1.0,
            setup_complexity=5,
            code_readability_score=8.0,
            error_rate=0.05,
        )
        
        assert result.framework == "Test Framework"
        assert result.task_name == "Test Task"
        assert result.lines_of_code == 10
        assert result.model_accuracy == 0.95

    def test_benchmark_result_to_dict(self):
        result = BenchmarkResult(
            framework="Test",
            task_name="Task",
            lines_of_code=10,
            development_time_seconds=5.0,
            training_time_seconds=30.0,
            inference_time_ms=2.5,
            peak_memory_mb=100.0,
            model_accuracy=0.95,
            model_size_mb=5.0,
            parameters_count=1000,
            compilation_time_seconds=1.0,
            setup_complexity=5,
            code_readability_score=8.0,
            error_rate=0.05,
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["framework"] == "Test"
        assert result_dict["lines_of_code"] == 10


class TestBenchmarkRunner:
    def test_runner_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir, verbose=False)
            assert runner.output_dir == Path(tmpdir)
            assert runner.verbose is False
            assert runner.results == []

    def test_save_and_load_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir, verbose=False)
            
            result = BenchmarkResult(
                framework="Test",
                task_name="Task",
                lines_of_code=10,
                development_time_seconds=5.0,
                training_time_seconds=30.0,
                inference_time_ms=2.5,
                peak_memory_mb=100.0,
                model_accuracy=0.95,
                model_size_mb=5.0,
                parameters_count=1000,
                compilation_time_seconds=1.0,
                setup_complexity=5,
                code_readability_score=8.0,
                error_rate=0.05,
            )
            
            runner.results.append(result)
            output_file = runner.save_results()
            
            assert output_file.exists()
            
            loaded_results = runner.load_results(str(output_file))
            assert len(loaded_results) == 1
            assert loaded_results[0].framework == "Test"

    def test_compare_frameworks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir, verbose=False)
            
            result1 = BenchmarkResult(
                framework="Framework1",
                task_name="Task",
                lines_of_code=10,
                development_time_seconds=5.0,
                training_time_seconds=30.0,
                inference_time_ms=2.5,
                peak_memory_mb=100.0,
                model_accuracy=0.95,
                model_size_mb=5.0,
                parameters_count=1000,
                compilation_time_seconds=1.0,
                setup_complexity=5,
                code_readability_score=8.0,
                error_rate=0.05,
            )
            
            result2 = BenchmarkResult(
                framework="Framework2",
                task_name="Task",
                lines_of_code=20,
                development_time_seconds=10.0,
                training_time_seconds=35.0,
                inference_time_ms=3.0,
                peak_memory_mb=120.0,
                model_accuracy=0.93,
                model_size_mb=6.0,
                parameters_count=1200,
                compilation_time_seconds=1.5,
                setup_complexity=8,
                code_readability_score=7.0,
                error_rate=0.07,
            )
            
            runner.results = [result1, result2]
            comparison = runner.compare_frameworks()
            
            assert "Framework1" in comparison
            assert "Framework2" in comparison
            assert comparison["Framework1"]["avg_lines_of_code"] == 10
            assert comparison["Framework2"]["avg_lines_of_code"] == 20


@pytest.mark.slow
class TestBenchmarkIntegration:
    @pytest.mark.skipif(
        not pytest.importorskip("tensorflow", reason="TensorFlow not installed"),
        reason="TensorFlow required for integration tests"
    )
    def test_run_neural_dsl_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir, verbose=False)
            impl = NeuralDSLImplementation()
            
            result = runner.run_benchmark(
                framework_impl=impl,
                task_name="test_task",
                dataset="mnist",
                epochs=1,
                batch_size=32,
                num_inference_samples=10,
            )
            
            assert result.framework == "Neural DSL"
            assert result.lines_of_code > 0
            assert result.model_accuracy >= 0

    @pytest.mark.skipif(
        not pytest.importorskip("tensorflow", reason="TensorFlow not installed"),
        reason="TensorFlow required for integration tests"
    )
    def test_run_keras_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir, verbose=False)
            impl = KerasImplementation()
            
            result = runner.run_benchmark(
                framework_impl=impl,
                task_name="test_task",
                dataset="mnist",
                epochs=1,
                batch_size=32,
                num_inference_samples=10,
            )
            
            assert result.framework == "Keras"
            assert result.lines_of_code > 0
            assert result.model_accuracy >= 0
