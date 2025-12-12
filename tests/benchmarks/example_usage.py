"""
Example usage of the Neural DSL benchmarking suite.

This script demonstrates how to:
1. Run benchmarks programmatically
2. Access and analyze results
3. Generate custom reports
"""

from tests.benchmarks import BenchmarkSuite, BenchmarkRunner, get_benchmark_models


def basic_benchmark_example():
    """Basic example: Run a simple benchmark."""
    print("=" * 80)
    print("Example 1: Basic Benchmark")
    print("=" * 80)
    
    suite = BenchmarkSuite(output_dir='./example_results')
    
    results = suite.run_all_benchmarks(
        backends=['tensorflow'],
        models=['simple_mlp'],
        epochs=2
    )
    
    report_path = suite.generate_markdown_report('example_report.md')
    print(f"\nReport generated: {report_path}")


def detailed_benchmark_example():
    """Detailed example: Access specific metrics."""
    print("\n" + "=" * 80)
    print("Example 2: Detailed Analysis")
    print("=" * 80)
    
    suite = BenchmarkSuite(output_dir='./detailed_results')
    
    results = suite.run_all_benchmarks(
        backends=['tensorflow'],
        models=['simple_mlp'],
        epochs=2
    )
    
    for result in results:
        comp = result['comparison']
        
        print(f"\nModel: {comp['model_name']}")
        print(f"Backend: {comp['backend']}")
        
        print("\nCompilation Overhead:")
        print(f"  Parse time: {comp['overhead']['parse_time']:.4f}s")
        print(f"  Codegen time: {comp['overhead']['codegen_time']:.4f}s")
        print(f"  Total: {comp['overhead']['total_overhead']:.4f}s")
        
        print("\nTraining Performance:")
        neural_time = comp['training_time']['neural_dsl']
        native_time = comp['training_time']['native']
        overhead_pct = comp['training_time']['percentage']
        print(f"  Neural DSL: {neural_time:.4f}s")
        print(f"  Native: {native_time:.4f}s")
        print(f"  Overhead: {overhead_pct:+.2f}%")
        
        print("\nMemory Usage:")
        neural_mem = comp['memory']['neural_dsl_mb']
        native_mem = comp['memory']['native_mb']
        mem_overhead_pct = comp['memory']['percentage']
        print(f"  Neural DSL: {neural_mem:.2f} MB")
        print(f"  Native: {native_mem:.2f} MB")
        print(f"  Overhead: {mem_overhead_pct:+.2f}%")
        
        print("\nModel Performance:")
        neural_acc = comp['accuracy']['neural_dsl']
        native_acc = comp['accuracy']['native']
        acc_diff = comp['accuracy']['difference']
        print(f"  Neural DSL accuracy: {neural_acc:.4f}")
        print(f"  Native accuracy: {native_acc:.4f}")
        print(f"  Difference: {acc_diff:+.4f}")


def custom_benchmark_example():
    """Custom example: Use BenchmarkRunner directly."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Benchmark with BenchmarkRunner")
    print("=" * 80)
    
    runner = BenchmarkRunner(output_dir='./custom_results')
    models = get_benchmark_models()
    
    try:
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        dataset = ((x_train[:1000], y_train[:1000]), (x_test[:200], y_test[:200]))
        print("Loaded MNIST dataset (subset for quick testing)")
    except:
        import numpy as np
        x_train = np.random.rand(1000, 28, 28).astype('float32')
        y_train = np.random.randint(0, 10, 1000)
        x_test = np.random.rand(200, 28, 28).astype('float32')
        y_test = np.random.randint(0, 10, 200)
        dataset = ((x_train, y_train), (x_test, y_test))
        print("Using synthetic dataset")
    
    print("\nBenchmarking Neural DSL...")
    neural_results = runner.benchmark_neural_dsl(
        model_name='simple_mlp',
        dsl_code=models['simple_mlp']['neural_dsl'],
        backend='tensorflow',
        dataset=dataset,
        epochs=2
    )
    
    print(f"  Parse time: {neural_results['parse_time']:.4f}s")
    print(f"  Codegen time: {neural_results['codegen_time']:.4f}s")
    print(f"  Training time: {neural_results['training_time']:.4f}s")
    print(f"  Final accuracy: {neural_results['final_accuracy']:.4f}")
    
    print("\nBenchmarking Native TensorFlow...")
    native_results = runner.benchmark_native(
        model_name='simple_mlp',
        native_code=models['simple_mlp']['tensorflow'],
        backend='tensorflow',
        dataset=dataset,
        epochs=2
    )
    
    print(f"  Training time: {native_results['training_time']:.4f}s")
    print(f"  Final accuracy: {native_results['final_accuracy']:.4f}")
    
    print("\nComparing results...")
    comparison = runner.compare_results(neural_results, native_results)
    
    overhead = comparison['training_time']['percentage']
    print(f"  Neural DSL overhead: {overhead:+.2f}%")
    
    if abs(overhead) < 10:
        print("  ✓ Overhead is within acceptable range (< 10%)")
    else:
        print("  ⚠ Overhead is higher than expected")


def multi_backend_example():
    """Multi-backend example: Compare TensorFlow and PyTorch."""
    print("\n" + "=" * 80)
    print("Example 4: Multi-Backend Comparison")
    print("=" * 80)
    
    suite = BenchmarkSuite(output_dir='./multi_backend_results')
    
    results = suite.run_all_benchmarks(
        backends=['tensorflow', 'pytorch'],
        models=['simple_mlp'],
        epochs=2
    )
    
    print("\nResults Summary:")
    for result in results:
        comp = result['comparison']
        backend = comp['backend']
        overhead = comp['training_time']['percentage']
        accuracy = comp['accuracy']['neural_dsl']
        
        print(f"\n{backend.title()}:")
        print(f"  Training overhead: {overhead:+.2f}%")
        print(f"  Final accuracy: {accuracy:.4f}")


def all_models_example():
    """All models example: Benchmark all available models."""
    print("\n" + "=" * 80)
    print("Example 5: All Models Benchmark")
    print("=" * 80)
    
    suite = BenchmarkSuite(output_dir='./all_models_results')
    
    print("Running benchmarks for all models...")
    print("(This may take a few minutes)\n")
    
    results = suite.run_all_benchmarks(
        backends=['tensorflow'],
        epochs=2
    )
    
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    for result in results:
        comp = result['comparison']
        model = comp['model_name']
        overhead = comp['training_time']['percentage']
        mem_overhead = comp['memory']['percentage']
        accuracy = comp['accuracy']['neural_dsl']
        
        print(f"\n{model}:")
        print(f"  Training overhead: {overhead:+.2f}%")
        print(f"  Memory overhead: {mem_overhead:+.2f}%")
        print(f"  Accuracy: {accuracy:.4f}")
    
    report_path = suite.generate_markdown_report('all_models_report.md')
    json_path = suite.save_results_json('all_models_results.json')
    
    print(f"\nReports saved:")
    print(f"  Markdown: {report_path}")
    print(f"  JSON: {json_path}")


if __name__ == '__main__':
    import sys
    
    examples = {
        '1': ('Basic benchmark', basic_benchmark_example),
        '2': ('Detailed analysis', detailed_benchmark_example),
        '3': ('Custom benchmark', custom_benchmark_example),
        '4': ('Multi-backend', multi_backend_example),
        '5': ('All models', all_models_example),
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        name, func = examples[sys.argv[1]]
        print(f"\nRunning: {name}\n")
        func()
    else:
        print("Neural DSL Benchmarking Examples")
        print("=" * 80)
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <number>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} 1")
        print("\nOr run all examples:")
        print(f"  python {sys.argv[0]} all")
        
        if len(sys.argv) > 1 and sys.argv[1] == 'all':
            print("\n" + "=" * 80)
            print("Running all examples...")
            print("=" * 80)
            for _, func in examples.values():
                try:
                    func()
                except Exception as e:
                    print(f"\nError: {e}")
                    print("Continuing with next example...\n")
