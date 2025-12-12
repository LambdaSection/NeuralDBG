# Neural DSL Benchmarking Suite

Comprehensive benchmarking suite comparing Neural DSL performance against raw TensorFlow and PyTorch implementations.

## Quick Start

```bash
# Install Neural DSL with full dependencies
pip install -e ".[full]"

# Run quick benchmark (2 epochs for testing)
python tests/benchmarks/run_benchmarks.py --quick --report

# Run full benchmark (5 epochs)
python tests/benchmarks/run_benchmarks.py --report --json

# Run specific backends and models
python tests/benchmarks/run_benchmarks.py \
    --backends tensorflow \
    --models simple_mlp,cnn \
    --epochs 10 \
    --report
```

## Structure

```
tests/benchmarks/
├── __init__.py              # Package exports
├── README.md                # This file
├── models.py                # Benchmark model definitions
├── benchmark_runner.py      # Core benchmarking engine
├── benchmark_suite.py       # Orchestration and reporting
├── test_benchmarks.py       # Unit tests for benchmarks
└── run_benchmarks.py        # Standalone CLI script
```

## What's Measured

### 1. Training Speed
- Time to complete fixed number of epochs
- Pure training time (excluding data loading)
- Total time including compilation overhead

### 2. Memory Usage
- Peak memory consumption during training
- Memory difference between Neural DSL and native
- Percentage overhead

### 3. Model Performance
- Final test accuracy
- Final test loss
- Training convergence

### 4. Compilation Overhead
- DSL parsing time
- Code generation time
- Total compilation overhead

## Benchmark Models

### Simple MLP
Basic multi-layer perceptron:
- **Architecture**: Flatten → Dense(128) → Dropout(0.2) → Output(10)
- **Parameters**: ~101,000
- **Use case**: Simple classification baseline

### CNN
Convolutional neural network:
- **Architecture**: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Output(10)
- **Parameters**: ~1,200,000
- **Use case**: Image classification

### Deep MLP
Deep fully-connected network:
- **Architecture**: Flatten → Dense(512) → Dense(256) → Dense(128) → Output(10)
- **Parameters**: ~670,000
- **Use case**: Deep architecture performance

## Usage Examples

### Running All Benchmarks

```bash
# Default: all models, both backends, 5 epochs
python tests/benchmarks/run_benchmarks.py --report --json
```

### Specific Configuration

```bash
# TensorFlow only, CNN model, 10 epochs
python tests/benchmarks/run_benchmarks.py \
    --backends tensorflow \
    --models cnn \
    --epochs 10 \
    --output ./my_results \
    --report
```

### Quick Testing

```bash
# Fast benchmark for CI/testing (2 epochs)
python tests/benchmarks/run_benchmarks.py --quick --report
```

## Programmatic API

```python
from tests.benchmarks import BenchmarkSuite

# Create suite
suite = BenchmarkSuite(output_dir='./my_benchmarks')

# Run benchmarks
results = suite.run_all_benchmarks(
    backends=['tensorflow', 'pytorch'],
    models=['simple_mlp', 'cnn'],
    epochs=5
)

# Generate reports
report_path = suite.generate_markdown_report('benchmark_report.md')
json_path = suite.save_results_json('benchmark_results.json')

# Access results
for result in results:
    comp = result['comparison']
    print(f"Model: {comp['model_name']}")
    print(f"Training time overhead: {comp['training_time']['percentage']:.2f}%")
    print(f"Memory overhead: {comp['memory']['percentage']:.2f}%")
```

## Using BenchmarkRunner Directly

```python
from tests.benchmarks import BenchmarkRunner, get_benchmark_models

runner = BenchmarkRunner(output_dir='./results')
models = get_benchmark_models()

# Load dataset
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
dataset = ((x_train, y_train), (x_test, y_test))

# Benchmark Neural DSL
neural_results = runner.benchmark_neural_dsl(
    model_name='simple_mlp',
    dsl_code=models['simple_mlp']['neural_dsl'],
    backend='tensorflow',
    dataset=dataset,
    epochs=5
)

# Benchmark native implementation
native_results = runner.benchmark_native(
    model_name='simple_mlp',
    native_code=models['simple_mlp']['tensorflow'],
    backend='tensorflow',
    dataset=dataset,
    epochs=5
)

# Compare results
comparison = runner.compare_results(neural_results, native_results)
print(f"Overhead: {comparison['training_time']['percentage']:.2f}%")
```

## Test Suite

Run unit tests:

```bash
# Run all benchmark tests
pytest tests/benchmarks/test_benchmarks.py -v

# Run specific test
pytest tests/benchmarks/test_benchmarks.py::TestBenchmarks::test_simple_mlp_tensorflow -v

# Run with coverage
pytest tests/benchmarks/test_benchmarks.py --cov=tests.benchmarks --cov-report=html
```

## Output Files

After running benchmarks, you'll find:

```
benchmark_results/
├── benchmark_report.md      # Markdown report with tables
├── benchmark_results.json   # Raw JSON data
├── simple_mlp_tensorflow_neural.py   # Generated Neural DSL code
├── simple_mlp_tensorflow_native.py   # Native implementation
├── cnn_pytorch_neural.py
├── cnn_pytorch_native.py
└── ...
```

## Interpreting Results

### Compilation Overhead
- **Expected**: 0.01 - 0.2 seconds
- **Impact**: One-time cost, amortized over training
- **Negligible for**: Training runs > 10 seconds

### Training Performance
- **Expected**: ±2-5% difference from native
- **Sources**: Experiment tracking, shape validation
- **Mitigation**: Disable optional features in production

### Memory Usage
- **Expected**: ±3-7% difference from native
- **Sources**: Tracking metadata, validation cache
- **Mitigation**: Use `--no-tracking` flag

### Model Quality
- **Expected**: Identical accuracy (< 0.01 difference)
- **Reason**: Same architecture and hyperparameters
- **Variance**: Normal training randomness

## Performance Tips

### For Best Performance

```bash
# Disable experiment tracking
export NEURAL_DISABLE_TRACKING=1

# Disable shape validation
neural compile model.neural --no-validation

# Use optimized builds
export TF_ENABLE_ONEDNN_OPTS=1  # TensorFlow
export PYTORCH_JIT=1             # PyTorch
```

### For Accurate Benchmarks

```python
# Warm up before timing
for _ in range(3):
    model.fit(x_train[:100], y_train[:100], epochs=1, verbose=0)

# Use consistent random seeds
import numpy as np
import tensorflow as tf
import torch

np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# Measure multiple runs
times = []
for _ in range(5):
    start = time.time()
    model.fit(x_train, y_train, epochs=5, verbose=0)
    times.append(time.time() - start)
avg_time = np.mean(times)
std_time = np.std(times)
```

## Adding New Benchmarks

1. **Add model to `models.py`:**

```python
'my_model': {
    'neural_dsl': """network MyModel { ... }""",
    'tensorflow': """def create_model(): ...""",
    'pytorch': """class MyModel(nn.Module): ..."""
}
```

2. **Ensure equivalence:**
   - Same architecture
   - Same hyperparameters
   - Same initialization (where possible)

3. **Add tests:**

```python
def test_my_model_tensorflow(self):
    results = self.suite.run_all_benchmarks(
        backends=['tensorflow'],
        models=['my_model'],
        epochs=2
    )
    self.assertEqual(len(results), 1)
```

4. **Run and validate:**

```bash
pytest tests/benchmarks/test_benchmarks.py::TestBenchmarks::test_my_model_tensorflow -v
python tests/benchmarks/run_benchmarks.py --models my_model --report
```

## Continuous Integration

Benchmarks run automatically on:
- Pull requests (quick mode)
- Weekly (full benchmarks)
- Releases (comprehensive suite)

Results are tracked over time to detect performance regressions.

## Troubleshooting

### "Module not found" errors

```bash
pip install -e ".[full]"  # Install all dependencies
```

### Out of memory errors

```bash
# Reduce batch size in model definitions
# Or use smaller models for testing
python tests/benchmarks/run_benchmarks.py --models simple_mlp --quick
```

### Different results each run

```python
# Set random seeds for reproducibility
import numpy as np
import tensorflow as tf
import torch

np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
```

### Slow benchmarks

```bash
# Use quick mode for testing
python tests/benchmarks/run_benchmarks.py --quick

# Or reduce epochs
python tests/benchmarks/run_benchmarks.py --epochs 2
```

## Contributing

We welcome benchmark contributions:

1. Add new models that represent real-world use cases
2. Optimize benchmark infrastructure
3. Add new metrics or analysis
4. Improve documentation

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## References

- [Full benchmark documentation](../../docs/benchmarks.md)
- [Neural DSL documentation](../../README.md)
- [Code generation details](../../docs/features/code_generation.md)

## License

Same as Neural DSL project (see [LICENSE.md](../../LICENSE.md))
