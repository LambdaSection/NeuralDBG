"""
Test suite for benchmarking Neural DSL against raw TensorFlow/PyTorch.
"""
import unittest
import os
from pathlib import Path
from .benchmark_suite import BenchmarkSuite


class TestBenchmarks(unittest.TestCase):
    """Test benchmarking functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.suite = BenchmarkSuite()
        cls.output_dir = Path(cls.suite.runner.output_dir)
    
    def test_simple_mlp_tensorflow(self):
        """Test simple MLP benchmark on TensorFlow."""
        results = self.suite.run_all_benchmarks(
            backends=['tensorflow'],
            models=['simple_mlp'],
            epochs=2
        )
        
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertIn('neural_dsl', result)
        self.assertIn('native', result)
        self.assertIn('comparison', result)
        
        self.assertGreater(result['neural_dsl']['final_accuracy'], 0.0)
        self.assertGreater(result['native']['final_accuracy'], 0.0)
    
    def test_simple_mlp_pytorch(self):
        """Test simple MLP benchmark on PyTorch."""
        results = self.suite.run_all_benchmarks(
            backends=['pytorch'],
            models=['simple_mlp'],
            epochs=2
        )
        
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertIn('neural_dsl', result)
        self.assertIn('native', result)
        self.assertIn('comparison', result)
        
        self.assertGreater(result['neural_dsl']['final_accuracy'], 0.0)
        self.assertGreater(result['native']['final_accuracy'], 0.0)
    
    def test_cnn_tensorflow(self):
        """Test CNN benchmark on TensorFlow."""
        results = self.suite.run_all_benchmarks(
            backends=['tensorflow'],
            models=['cnn'],
            epochs=2
        )
        
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertIn('comparison', result)
        comp = result['comparison']
        
        self.assertIn('overhead', comp)
        self.assertGreater(comp['overhead']['parse_time'], 0)
        self.assertGreater(comp['overhead']['codegen_time'], 0)
    
    def test_deep_mlp_pytorch(self):
        """Test deep MLP benchmark on PyTorch."""
        results = self.suite.run_all_benchmarks(
            backends=['pytorch'],
            models=['deep_mlp'],
            epochs=2
        )
        
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertIn('comparison', result)
        comp = result['comparison']
        
        self.assertIn('training_time', comp)
        self.assertGreater(comp['training_time']['neural_dsl'], 0)
        self.assertGreater(comp['training_time']['native'], 0)
    
    def test_all_models_tensorflow(self):
        """Test all models on TensorFlow backend."""
        results = self.suite.run_all_benchmarks(
            backends=['tensorflow'],
            epochs=2
        )
        
        self.assertGreaterEqual(len(results), 3)
        
        for result in results:
            self.assertIn('neural_dsl', result)
            self.assertIn('native', result)
            self.assertIn('comparison', result)
    
    def test_all_models_pytorch(self):
        """Test all models on PyTorch backend."""
        results = self.suite.run_all_benchmarks(
            backends=['pytorch'],
            epochs=2
        )
        
        self.assertGreaterEqual(len(results), 3)
        
        for result in results:
            self.assertIn('neural_dsl', result)
            self.assertIn('native', result)
            self.assertIn('comparison', result)
    
    def test_comparison_metrics(self):
        """Test that comparison metrics are properly calculated."""
        results = self.suite.run_all_benchmarks(
            backends=['tensorflow'],
            models=['simple_mlp'],
            epochs=2
        )
        
        comp = results[0]['comparison']
        
        self.assertIn('overhead', comp)
        self.assertIn('training_time', comp)
        self.assertIn('memory', comp)
        self.assertIn('accuracy', comp)
        self.assertIn('loss', comp)
        
        self.assertIn('percentage', comp['training_time'])
        self.assertIn('difference', comp['training_time'])
        
        self.assertIn('percentage', comp['memory'])
        self.assertIn('difference_mb', comp['memory'])
    
    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        self.suite.run_all_benchmarks(
            backends=['tensorflow'],
            models=['simple_mlp'],
            epochs=2
        )
        
        report_path = self.suite.generate_markdown_report('test_report.md')
        
        self.assertTrue(os.path.exists(report_path))
        
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn('# Neural DSL Benchmark Results', content)
        self.assertIn('## Summary', content)
        self.assertIn('## Detailed Results', content)
        self.assertIn('## Aggregate Statistics', content)
    
    def test_save_results_json(self):
        """Test JSON results saving."""
        self.suite.run_all_benchmarks(
            backends=['tensorflow'],
            models=['simple_mlp'],
            epochs=2
        )
        
        json_path = self.suite.save_results_json('test_results.json')
        
        self.assertTrue(os.path.exists(json_path))
        
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)


class TestBenchmarkRunner(unittest.TestCase):
    """Test BenchmarkRunner functionality."""
    
    def setUp(self):
        """Set up test environment."""
        from .benchmark_runner import BenchmarkRunner
        self.runner = BenchmarkRunner()
    
    def test_benchmark_neural_dsl_tensorflow(self):
        """Test Neural DSL benchmarking for TensorFlow."""
        from .models import get_benchmark_models
        models = get_benchmark_models()
        
        dataset = self._get_small_dataset()
        
        result = self.runner.benchmark_neural_dsl(
            model_name='simple_mlp',
            dsl_code=models['simple_mlp']['neural_dsl'],
            backend='tensorflow',
            dataset=dataset,
            epochs=1
        )
        
        self.assertIn('model_name', result)
        self.assertIn('backend', result)
        self.assertIn('framework', result)
        self.assertIn('parse_time', result)
        self.assertIn('codegen_time', result)
        self.assertIn('training_time', result)
        self.assertIn('memory_used_mb', result)
        self.assertIn('final_accuracy', result)
        self.assertIn('final_loss', result)
        
        self.assertEqual(result['model_name'], 'simple_mlp')
        self.assertEqual(result['backend'], 'tensorflow')
        self.assertEqual(result['framework'], 'neural_dsl')
    
    def test_benchmark_native_pytorch(self):
        """Test native PyTorch benchmarking."""
        from .models import get_benchmark_models
        models = get_benchmark_models()
        
        dataset = self._get_small_dataset()
        
        result = self.runner.benchmark_native(
            model_name='simple_mlp',
            native_code=models['simple_mlp']['pytorch'],
            backend='pytorch',
            dataset=dataset,
            epochs=1
        )
        
        self.assertIn('model_name', result)
        self.assertIn('backend', result)
        self.assertIn('framework', result)
        self.assertIn('training_time', result)
        self.assertIn('memory_used_mb', result)
        self.assertIn('final_accuracy', result)
        self.assertIn('final_loss', result)
        
        self.assertEqual(result['framework'], 'native')
        self.assertEqual(result['total_overhead'], 0.0)
    
    def test_compare_results(self):
        """Test results comparison."""
        neural_results = {
            'model_name': 'test_model',
            'backend': 'tensorflow',
            'parse_time': 0.1,
            'codegen_time': 0.2,
            'total_overhead': 0.3,
            'training_time': 10.5,
            'memory_used_mb': 150.0,
            'final_accuracy': 0.95,
            'final_loss': 0.15
        }
        
        native_results = {
            'model_name': 'test_model',
            'backend': 'tensorflow',
            'training_time': 10.0,
            'memory_used_mb': 140.0,
            'final_accuracy': 0.94,
            'final_loss': 0.16
        }
        
        comparison = self.runner.compare_results(neural_results, native_results)
        
        self.assertIn('model_name', comparison)
        self.assertIn('overhead', comparison)
        self.assertIn('training_time', comparison)
        self.assertIn('memory', comparison)
        self.assertIn('accuracy', comparison)
        self.assertIn('loss', comparison)
        
        self.assertAlmostEqual(comparison['training_time']['difference'], 0.5, places=1)
        self.assertAlmostEqual(comparison['memory']['difference_mb'], 10.0, places=1)
    
    def _get_small_dataset(self):
        """Get a small synthetic dataset for testing."""
        import numpy as np
        x_train = np.random.rand(100, 28, 28).astype('float32')
        y_train = np.random.randint(0, 10, 100)
        x_test = np.random.rand(20, 28, 28).astype('float32')
        y_test = np.random.randint(0, 10, 20)
        return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    unittest.main()
