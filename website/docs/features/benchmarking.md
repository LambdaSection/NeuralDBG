# Benchmarking Guide

This guide explains how to run and interpret Neural DSL benchmarks, and how to use benchmarking to validate your model performance.

## Overview

Neural DSL includes a comprehensive benchmarking suite that:

- Compares against industry-standard frameworks (TensorFlow, PyTorch, Keras, etc.)
- Measures code quality, development time, and runtime performance
- Generates publication-ready visualizations and reports
- Provides reproducible scripts for validation

## Quick Start

### 1. Install Dependencies

```bash
# Install Neural DSL with benchmarking support
pip install -e ".[full]"

# Install optional competing frameworks
pip install -r neural/benchmarks/requirements.txt
```

### 2. Run Quick Benchmark

```bash
# Interactive quick benchmark (2-3 minutes)
python neural/benchmarks/quick_start.py
```

This runs a simple comparison between Neural DSL and Keras, perfect for demos.

### 3. View Results

The quick start shows results directly in the terminal:
- Lines of code comparison
- Development time
- Model accuracy
- Training performance

## Running Full Benchmarks

### Comprehensive Suite

Run the full benchmark suite comparing all frameworks:

```bash
python neural/benchmarks/run_benchmarks.py
```

This benchmarks:
- Neural DSL
- Keras
- Raw TensorFlow
- PyTorch Lightning
- Raw PyTorch
- Fast.ai (if installed)
- Ludwig (if installed)

### Selective Benchmarks

Benchmark specific frameworks:

```bash
# Compare against raw implementations
python neural/benchmarks/run_benchmarks.py --frameworks neural raw-tensorflow raw-pytorch

# Compare against high-level frameworks
python neural/benchmarks/run_benchmarks.py --frameworks neural keras pytorch-lightning

# Just Neural DSL vs Keras (fastest)
python neural/benchmarks/run_benchmarks.py --frameworks neural keras
```

### Custom Parameters

Adjust training parameters:

```bash
# More epochs for better accuracy comparison
python neural/benchmarks/run_benchmarks.py --epochs 10

# Larger batch size for speed comparison
python neural/benchmarks/run_benchmarks.py --batch-size 128

# Save to custom directory
python neural/benchmarks/run_benchmarks.py --output-dir my_results --report-dir my_reports
```

## Understanding Results

### Key Metrics

**Lines of Code (LOC)**
- Measures code conciseness
- Neural DSL typically achieves 60-75% reduction
- Lower is better

**Development Time**
- Time from starting to write code to having a runnable model
- Includes setup and compilation
- Neural DSL is typically 3-5x faster

**Training Time**
- Wall-clock time for training
- Neural DSL has zero overhead (compiles to native code)
- Within 5% of hand-written implementations

**Inference Time**
- Average prediction latency per sample
- Critical for production deployment
- Neural DSL matches raw implementations

**Model Accuracy**
- Test set performance
- Neural DSL produces mathematically equivalent models
- Accuracy should be within 0.01 of other frameworks

**Code Readability**
- Subjective score based on complexity
- Neural DSL scores 8-9/10
- Raw implementations score 5-6/10

### Reading Reports

After running benchmarks, open the HTML report:

```bash
# Find latest report
ls -lt benchmark_reports/

# Open in browser (macOS)
open benchmark_reports/neural_dsl_benchmark_*/index.html

# Open in browser (Linux)
xdg-open benchmark_reports/neural_dsl_benchmark_*/index.html

# Open in browser (Windows)
start benchmark_reports/neural_dsl_benchmark_*/index.html
```

The report includes:
- Executive summary with key findings
- Comparison charts for each metric
- Detailed results table
- Reproducibility instructions

## Reproducing Benchmarks

Each benchmark report includes a `reproduce.py` script:

```bash
cd benchmark_reports/neural_dsl_benchmark_TIMESTAMP/
python reproduce.py
```

This re-runs the exact same benchmarks with identical parameters.

## Publishing Results

### Generate Website Content

Publish benchmarks to website documentation:

```bash
# Run benchmarks and update website
python neural/benchmarks/publish_to_website.py --run-benchmarks

# Just update website from latest results
python neural/benchmarks/publish_to_website.py
```

This updates:
- `website/docs/benchmarks.md` - Main benchmark documentation
- `website/docs/benchmark_summary.md` - Quick summary
- `website/docs/assets/benchmarks/` - Visualization images
- `website/static/benchmarks/latest/` - Interactive reports

### Custom Visualizations

Generate specific visualizations:

```bash
# Generate all standard plots
python neural/benchmarks/visualization.py benchmark_results/benchmark_results_*.json

# Custom output directory
python neural/benchmarks/visualization.py benchmark_results/benchmark_results_*.json --output-dir my_plots
```

## Advanced Usage

### Programmatic Benchmarking

Use the benchmark API in your own scripts:

```python
from neural.benchmarks.benchmark_runner import BenchmarkRunner
from neural.benchmarks.framework_implementations import (
    NeuralDSLImplementation,
    KerasImplementation,
)
from neural.benchmarks.report_generator import ReportGenerator

# Setup frameworks
frameworks = [
    NeuralDSLImplementation(),
    KerasImplementation(),
]

# Define tasks
tasks = [{
    "name": "MyTask",
    "dataset": "mnist",
    "epochs": 5,
    "batch_size": 32,
}]

# Run benchmarks
runner = BenchmarkRunner(output_dir="my_results")
results = runner.run_all_benchmarks(frameworks, tasks)

# Generate report
report_gen = ReportGenerator(output_dir="my_reports")
report_path = report_gen.generate_report([r.to_dict() for r in results])
```

### Custom Metrics

Collect additional metrics:

```python
from neural.benchmarks.metrics_collector import (
    MetricsCollector,
    ResourceMonitor,
    CodeMetrics,
)

# Initialize collector
collector = MetricsCollector()

# Start resource monitoring
collector.start_monitoring()

# Your training code here
# ...

# Stop and get results
resource_stats = collector.stop_monitoring()

print(f"Peak memory: {resource_stats['peak_memory_mb']:.2f} MB")
print(f"Avg CPU: {resource_stats['avg_cpu_percent']:.1f}%")
```

### Custom Framework Implementation

Benchmark your own framework:

```python
from neural.benchmarks.framework_implementations import FrameworkImplementation

class MyFrameworkImplementation(FrameworkImplementation):
    def __init__(self):
        super().__init__("My Framework")
    
    def setup(self):
        self.code_content = """
        # Your framework code here
        """
    
    def build_model(self):
        # Build your model
        self.model = ...
    
    def train(self, dataset, epochs, batch_size):
        # Train and return metrics
        return {
            "training_time": ...,
            "accuracy": ...,
            "val_accuracy": ...,
            "val_loss": ...,
            "training_loss": ...,
            "peak_memory_mb": 0,
            "error_rate": ...,
        }
    
    def predict_single(self):
        # Make prediction
        return self.model.predict(...)
    
    def _save_model(self, path):
        # Save model
        ...
    
    def get_parameter_count(self):
        # Return parameter count
        return ...

# Use in benchmarks
runner = BenchmarkRunner()
result = runner.run_benchmark(
    framework_impl=MyFrameworkImplementation(),
    task_name="Test",
    dataset="mnist",
)
```

## Best Practices

### Fair Comparisons

1. **Identical Hardware**: Run all benchmarks on the same machine
2. **Same Model Architecture**: Use equivalent layer configurations
3. **Same Hyperparameters**: Learning rate, batch size, epochs
4. **Multiple Runs**: Average over 5-10 runs for stable results
5. **Warm-up**: Run once to load data/models before measuring

### Interpreting Differences

**Small Differences (<10%)**
- Training/inference time differences <10% are noise
- Accuracy differences <0.01 are statistically insignificant
- Focus on trends, not individual runs

**Large Differences (>20%)**
- Verify implementations are truly equivalent
- Check for framework-specific optimizations
- Consider hardware/system load factors

### Reporting Results

When publishing benchmarks:

1. **Include System Info**: CPU, GPU, RAM, OS
2. **Document Parameters**: Epochs, batch size, learning rate
3. **Provide Reproducibility**: Share code and data
4. **Show Variance**: Error bars or confidence intervals
5. **Be Honest**: Report both strengths and limitations

## Troubleshooting

### Common Issues

**"Framework not available"**
```bash
# Install missing framework
pip install tensorflow  # or torch, pytorch-lightning, etc.
```

**"Out of memory"**
```bash
# Reduce batch size
python neural/benchmarks/run_benchmarks.py --batch-size 16

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
```

**"Results vary significantly"**
- Ensure deterministic mode: Set random seeds
- Close other applications
- Run multiple times and average
- Check for thermal throttling

**"Compilation too slow"**
- Compilation is one-time cost
- Ensure you're measuring training time separately
- Use `--skip-plots` to speed up report generation

### Getting Help

- **Documentation**: [/docs/](/docs/)
- **GitHub Issues**: [Report bugs](https://github.com/your-org/neural-dsl/issues)
- **Community**: [Discord](#) | [Forum](#)

## Contributing

Help improve our benchmarks:

1. **Add Frameworks**: Implement new framework comparisons
2. **Add Metrics**: Track additional performance indicators
3. **Add Models**: Expand beyond MNIST
4. **Improve Fairness**: Suggest better comparison methodologies

See [CONTRIBUTING.md](/CONTRIBUTING.md) for details.

## See Also

- [Full Benchmark Results](/docs/benchmarks.md)
- [Performance Optimization](/docs/features/optimization.md)
- [Multi-Backend Support](/docs/features/backends.md)
