# Benchmarking Suite Implementation Details

This document describes the implementation of the Neural DSL benchmarking suite.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     BenchmarkSuite                           │
│  - Orchestrates all benchmarks                              │
│  - Manages multiple models and backends                     │
│  - Generates reports                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ uses
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   BenchmarkRunner                            │
│  - Executes individual benchmarks                           │
│  - Measures performance metrics                             │
│  - Compares results                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ uses
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Model Definitions (models.py)                 │
│  - Neural DSL models                                        │
│  - Equivalent TensorFlow implementations                    │
│  - Equivalent PyTorch implementations                       │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. models.py

Defines benchmark models in three formats:
- **Neural DSL**: High-level declarative syntax
- **TensorFlow**: Keras functional API implementation
- **PyTorch**: nn.Module implementation

Each model includes:
- Identical architecture
- Same hyperparameters
- Equivalent layers and activations

### 2. benchmark_runner.py

Core benchmarking engine that:

**For Neural DSL models:**
1. Parses DSL code to AST
2. Generates target framework code
3. Executes generated code
4. Measures compilation and training time
5. Tracks memory usage
6. Records model performance

**For native models:**
1. Executes native implementation directly
2. Measures training time
3. Tracks memory usage
4. Records model performance

**Metrics collected:**
- Parse time (DSL only)
- Code generation time (DSL only)
- Training time
- Memory usage (peak and delta)
- Final accuracy
- Final loss

### 3. benchmark_suite.py

Orchestration layer that:
1. Runs multiple benchmarks in sequence
2. Aggregates results
3. Generates comparison reports
4. Produces markdown and JSON outputs

**Report generation:**
- Detailed per-model results
- Aggregate statistics
- Performance comparisons
- Overhead analysis

### 4. test_benchmarks.py

Comprehensive test suite:
- Unit tests for BenchmarkRunner
- Integration tests for BenchmarkSuite
- Model validation tests
- Report generation tests

### 5. run_benchmarks.py

Command-line interface:
- Configurable backends
- Configurable models
- Configurable epochs
- Output directory selection
- Report generation options

## Implementation Details

### Metric Collection

#### Timing
```python
start = time.time()
# Execute operation
elapsed = time.time() - start
```

#### Memory
```python
import psutil
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024  # MB
# Execute operation
mem_after = process.memory_info().rss / 1024 / 1024  # MB
memory_used = mem_after - mem_before
```

#### Model Performance
```python
# After training
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
```

### Code Execution

Generated code is executed using Python's `exec()`:
```python
namespace = {}
exec(generated_code, namespace)
model = namespace['model']
```

This ensures:
- Clean execution environment
- No namespace pollution
- Proper isolation

### Comparison Logic

```python
comparison = {
    'training_time': {
        'neural_dsl': neural_time,
        'native': native_time,
        'difference': neural_time - native_time,
        'percentage': ((neural_time - native_time) / native_time) * 100
    },
    # Similar for other metrics
}
```

## Design Decisions

### Why Three Model Formats?

**Neural DSL**: The subject of benchmarking  
**TensorFlow**: Most popular framework, reference implementation  
**PyTorch**: Popular alternative, validates cross-framework consistency

### Why MNIST?

- Small dataset, fast training
- Well-understood baseline
- Available in both TF and PyTorch
- Representative of typical workflows

### Why Multiple Models?

**Simple MLP**: Baseline, minimal overhead  
**CNN**: Realistic architecture, convolutional layers  
**Deep MLP**: Tests scaling with depth

### Accuracy Tolerance

Models are considered equivalent if:
- Accuracy difference < 0.01 (1%)
- Loss difference < 0.1

This accounts for:
- Random initialization
- Training variance
- Numerical precision

## Extensibility

### Adding New Models

```python
# In models.py
'my_model': {
    'neural_dsl': """network MyModel { ... }""",
    'tensorflow': """def create_model(): ...""",
    'pytorch': """class MyModel(nn.Module): ..."""
}
```

Requirements:
- Equivalent architectures
- Same hyperparameters
- Same input/output shapes

### Adding New Metrics

```python
# In benchmark_runner.py
def _execute_training(self, ...):
    # Existing metrics
    metrics['training_time'] = train_time
    metrics['memory_used_mb'] = mem_delta
    
    # Add new metric
    metrics['new_metric'] = calculate_new_metric()
    return metrics
```

### Adding New Backends

```python
# In benchmark_runner.py
def _execute_training(self, code_file, backend, ...):
    if backend == 'tensorflow':
        return self._train_tensorflow(...)
    elif backend == 'pytorch':
        return self._train_pytorch(...)
    elif backend == 'new_backend':
        return self._train_new_backend(...)
```

## Testing Strategy

### Unit Tests
- Individual component functionality
- Mock data for fast execution
- Isolated testing

### Integration Tests
- End-to-end workflows
- Real data (small subsets)
- Component interaction

### Smoke Tests
- Quick validation
- Reduced epochs
- Fast feedback

## Performance Considerations

### Optimization Techniques

1. **Small datasets for testing**: Use subsets to reduce test time
2. **Caching**: Cache parsed models and generated code
3. **Parallel execution**: Could be added for multiple benchmarks
4. **Warm-up runs**: Excluded from timing to ensure fair comparison

### Memory Management

- Explicit garbage collection between benchmarks
- Process isolation option (not implemented yet)
- Memory profiling tools integration

## Known Limitations

1. **Single-process execution**: Sequential benchmarks only
2. **CPU/GPU differences**: Results vary by hardware
3. **Dataset dependency**: Currently MNIST-focused
4. **No distributed training**: Single-machine only

## Future Enhancements

### Planned Features

1. **More models**: Transformers, RNNs, GANs
2. **More datasets**: CIFAR-10, ImageNet subset
3. **More backends**: ONNX, JAX
4. **Distributed benchmarks**: Multi-GPU, multi-node
5. **Inference benchmarks**: Not just training
6. **Energy consumption**: Power usage metrics
7. **Compilation caching**: Speed up repeated benchmarks
8. **Interactive dashboard**: Real-time results visualization

### Community Contributions

We welcome:
- New model implementations
- Additional backends
- Performance optimizations
- Bug fixes
- Documentation improvements

## References

### Internal
- `neural/parser/parser.py` - DSL parsing
- `neural/code_generation/code_generator.py` - Code generation
- `neural/shape_propagation/shape_propagator.py` - Shape validation

### External
- [TensorFlow Performance Guide](https://www.tensorflow.org/guide/profiler)
- [PyTorch Benchmarking](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [Python Profiling](https://docs.python.org/3/library/profile.html)

## Changelog

### Version 1.0.0 (Current)
- Initial implementation
- Three benchmark models (simple_mlp, cnn, deep_mlp)
- TensorFlow and PyTorch backends
- Comprehensive test suite
- Markdown and JSON report generation
- Command-line interface

---

*For usage instructions, see [README.md](README.md)*  
*For benchmark results, see [../../docs/benchmarks.md](../../docs/benchmarks.md)*
