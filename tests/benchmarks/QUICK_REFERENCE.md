# Neural DSL Benchmarking - Quick Reference

## Installation

```bash
pip install -e ".[full]"
```

## Verify Setup

```bash
python tests/benchmarks/verify_setup.py
```

## Run Benchmarks

### Quick Test (2 epochs)
```bash
python tests/benchmarks/run_benchmarks.py --quick --report
```

### Full Benchmark (5 epochs)
```bash
python tests/benchmarks/run_benchmarks.py --report --json
```

### Custom Configuration
```bash
python tests/benchmarks/run_benchmarks.py \
    --backends tensorflow,pytorch \
    --models simple_mlp,cnn \
    --epochs 10 \
    --output ./my_results \
    --report --json
```

## Run Tests

```bash
# All tests
pytest tests/benchmarks/test_benchmarks.py -v

# Specific test
pytest tests/benchmarks/test_benchmarks.py::TestBenchmarks::test_simple_mlp_tensorflow -v

# With coverage
pytest tests/benchmarks/test_benchmarks.py --cov=tests.benchmarks
```

## Example Usage

```bash
# View all examples
python tests/benchmarks/example_usage.py

# Run specific example
python tests/benchmarks/example_usage.py 1

# Run all examples
python tests/benchmarks/example_usage.py all
```

## Programmatic API

### Basic Usage
```python
from tests.benchmarks import BenchmarkSuite

suite = BenchmarkSuite()
results = suite.run_all_benchmarks(
    backends=['tensorflow'],
    models=['simple_mlp'],
    epochs=5
)
suite.generate_markdown_report()
```

### Advanced Usage
```python
from tests.benchmarks import BenchmarkRunner, get_benchmark_models

runner = BenchmarkRunner()
models = get_benchmark_models()

# Load dataset
import tensorflow as tf
dataset = tf.keras.datasets.mnist.load_data()

# Benchmark Neural DSL
neural_results = runner.benchmark_neural_dsl(
    model_name='simple_mlp',
    dsl_code=models['simple_mlp']['neural_dsl'],
    backend='tensorflow',
    dataset=dataset,
    epochs=5
)

# Benchmark native
native_results = runner.benchmark_native(
    model_name='simple_mlp',
    native_code=models['simple_mlp']['tensorflow'],
    backend='tensorflow',
    dataset=dataset,
    epochs=5
)

# Compare
comparison = runner.compare_results(neural_results, native_results)
print(f"Overhead: {comparison['training_time']['percentage']:.2f}%")
```

## Available Models

- **simple_mlp**: Basic fully-connected network (~101K params)
- **cnn**: Convolutional network (~1.2M params)
- **deep_mlp**: Deep fully-connected network (~670K params)

## Available Backends

- **tensorflow**: TensorFlow/Keras
- **pytorch**: PyTorch

## Output Files

After running benchmarks:

```
benchmark_results/
├── benchmark_report.md              # Markdown report
├── benchmark_results.json           # Raw JSON data
├── {model}_{backend}_neural.py      # Generated Neural DSL code
└── {model}_{backend}_native.py      # Native implementation
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backends` | Comma-separated backends | tensorflow,pytorch |
| `--models` | Comma-separated models | simple_mlp,cnn,deep_mlp |
| `--epochs` | Number of training epochs | 5 |
| `--output` | Output directory | ./benchmark_results |
| `--report` | Generate markdown report | False |
| `--json` | Save JSON results | False |
| `--quick` | Quick test (2 epochs) | False |

## Troubleshooting

### Import Errors
```bash
pip install -e ".[full]"
```

### Out of Memory
```bash
python tests/benchmarks/run_benchmarks.py --models simple_mlp --quick
```

### Different Results
```python
# Set seeds for reproducibility
import numpy as np
import tensorflow as tf
import torch

np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
```

## Documentation

- [README.md](README.md) - User guide
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical details
- [../../docs/benchmarks.md](../../docs/benchmarks.md) - Results documentation
- [example_usage.py](example_usage.py) - Code examples

## Help

```bash
# Command-line help
python tests/benchmarks/run_benchmarks.py --help

# Verify setup
python tests/benchmarks/verify_setup.py

# View examples
python tests/benchmarks/example_usage.py
```

## Quick Commands Cheatsheet

```bash
# Verify setup
python tests/benchmarks/verify_setup.py

# Quick benchmark
python tests/benchmarks/run_benchmarks.py --quick --report

# TensorFlow only
python tests/benchmarks/run_benchmarks.py --backends tensorflow --report

# PyTorch only
python tests/benchmarks/run_benchmarks.py --backends pytorch --report

# Single model
python tests/benchmarks/run_benchmarks.py --models simple_mlp --report

# Full benchmark with all outputs
python tests/benchmarks/run_benchmarks.py --report --json

# Custom epochs
python tests/benchmarks/run_benchmarks.py --epochs 10 --report

# Run tests
pytest tests/benchmarks/test_benchmarks.py -v

# Examples
python tests/benchmarks/example_usage.py 1
```

## Expected Results

| Metric | Expected Range |
|--------|---------------|
| Parse time | 0.001 - 0.01s |
| Codegen time | 0.01 - 0.1s |
| Training overhead | ±2-5% |
| Memory overhead | ±3-7% |
| Accuracy difference | < 0.01 |

## Performance Tips

```bash
# Disable tracking for best performance
export NEURAL_DISABLE_TRACKING=1

# Use optimized builds
export TF_ENABLE_ONEDNN_OPTS=1
export PYTORCH_JIT=1
```

---

*For detailed information, see [README.md](README.md)*
