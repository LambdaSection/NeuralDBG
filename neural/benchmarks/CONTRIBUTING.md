# Contributing to Neural DSL Benchmarks

Thank you for your interest in improving the Neural DSL benchmarking suite! This document provides guidelines for contributing new benchmarks, frameworks, and improvements.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Neural.git`
3. Create a branch: `git checkout -b feature/my-new-benchmark`
4. Make your changes
5. Test thoroughly
6. Submit a pull request

## Areas for Contribution

### 1. New Framework Implementations

To add support for a new ML framework:

```python
# neural/benchmarks/framework_implementations.py

class NewFrameworkImplementation(FrameworkImplementation):
    def __init__(self):
        super().__init__("NewFramework")
    
    def setup(self):
        # Define code that would be written
        self.code_content = '''
import newframework as nf

model = nf.Model([
    nf.layers.Conv2D(32, (3,3)),
    nf.layers.Dense(10),
])
'''
    
    def build_model(self):
        # Actually build the model
        import newframework as nf
        self.model = nf.Model([...])
    
    def train(self, dataset, epochs, batch_size):
        # Implement training
        start = time.time()
        history = self.model.fit(...)
        training_time = time.time() - start
        
        return {
            "training_time": training_time,
            "accuracy": ...,
            "val_accuracy": ...,
            "val_loss": ...,
            "peak_memory_mb": ...,
        }
    
    def predict_single(self):
        return self.model.predict(sample)
    
    def _save_model(self, path):
        self.model.save(path)
    
    def get_parameter_count(self):
        return self.model.count_params()
```

**Requirements:**
- Must implement all abstract methods
- Should use equivalent model architecture to existing frameworks
- Must track accurate metrics (time, memory, accuracy)
- Code should be well-documented

### 2. New Benchmark Tasks

Add new tasks to benchmark different model types:

```python
# In your benchmark script or config

tasks = [
    {
        "name": "CIFAR10_Classification",
        "dataset": "cifar10",
        "epochs": 10,
        "batch_size": 128,
    },
    {
        "name": "IMDB_Sentiment_Analysis",
        "dataset": "imdb",
        "epochs": 5,
        "batch_size": 32,
    },
]
```

**Requirements:**
- Task must be reproducible
- Dataset must be publicly available
- Should test different aspects (CV, NLP, etc.)
- Must document expected results

### 3. Enhanced Metrics

Add new metrics to collect:

```python
# neural/benchmarks/metrics_collector.py

class EnhancedMetricsCollector(MetricsCollector):
    def collect_gpu_metrics(self):
        # Use nvidia-smi or similar
        return {
            "gpu_utilization": ...,
            "gpu_memory_used": ...,
            "gpu_temperature": ...,
        }
    
    def collect_power_metrics(self):
        # Measure power consumption
        return {
            "power_draw_watts": ...,
            "energy_consumed_joules": ...,
        }
```

**Requirements:**
- Must work across different hardware
- Should fail gracefully if hardware not available
- Must be properly documented

### 4. Improved Visualizations

Enhance report generation with better charts:

```python
# neural/benchmarks/report_generator.py

def _generate_advanced_plots(self, results, output_dir):
    # Create interactive plotly charts
    # Add statistical significance tests
    # Generate comparison matrices
    # Create radar charts for multi-metric comparison
    pass
```

**Requirements:**
- Charts must be clear and informative
- Should work with matplotlib, plotly, or seaborn
- Must handle edge cases (missing data, etc.)

## Testing Guidelines

### Unit Tests

Add tests for new functionality:

```python
# tests/benchmarks/test_new_feature.py

import pytest
from neural.benchmarks import MyNewFeature

def test_new_feature():
    feature = MyNewFeature()
    result = feature.do_something()
    assert result == expected_value

@pytest.mark.slow
def test_integration():
    # Integration tests marked as slow
    pass
```

Run tests:
```bash
pytest tests/benchmarks/ -v
```

### Integration Tests

For framework implementations, add integration tests:

```python
@pytest.mark.slow
@pytest.mark.requires_framework("newframework")
def test_newframework_benchmark():
    impl = NewFrameworkImplementation()
    impl.setup()
    impl.build_model()
    result = impl.train("mnist", epochs=1, batch_size=32)
    assert result["accuracy"] > 0
```

## Code Style

Follow these conventions:

1. **PEP 8**: Use standard Python style
2. **Type Hints**: Add type hints to function signatures
3. **Docstrings**: Document all public functions
4. **Line Length**: Max 100 characters
5. **Imports**: Organize with isort

Format code:
```bash
python -m ruff format neural/benchmarks/
python -m ruff check neural/benchmarks/
```

## Documentation

Update documentation for new features:

1. **README.md**: Add to quick start if needed
2. **BENCHMARKS.md**: Document new metrics/frameworks
3. **Docstrings**: Add to all functions
4. **Examples**: Create example scripts

## Pull Request Process

1. **Update Tests**: Add tests for new functionality
2. **Update Docs**: Document changes
3. **Run Tests**: Ensure all tests pass
4. **Run Benchmarks**: Verify benchmarks work end-to-end
5. **Format Code**: Run ruff/pylint
6. **Create PR**: Include description of changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New framework implementation
- [ ] New benchmark task
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Benchmarks run successfully

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

## Performance Considerations

When adding benchmarks:

1. **Memory Efficiency**: Avoid loading entire datasets into memory
2. **Time Efficiency**: Use appropriate batch sizes and epochs
3. **Resource Cleanup**: Always clean up temporary files
4. **Graceful Degradation**: Handle missing dependencies

Example:
```python
def cleanup(self):
    # Clean up resources
    for temp_file in self.temp_files:
        try:
            os.remove(temp_file)
        except Exception:
            pass
```

## Common Issues

### Framework Not Installed

Handle gracefully:
```python
try:
    import newframework
except ImportError:
    raise ImportError(
        "NewFramework not installed. Install with: "
        "pip install newframework"
    )
```

### Dataset Download Failures

Provide clear error messages:
```python
try:
    dataset = load_dataset(name)
except Exception as e:
    raise RuntimeError(
        f"Failed to load dataset {name}. "
        f"Please check your internet connection and try again. "
        f"Error: {e}"
    )
```

## Questions?

- Open an issue: https://github.com/Lemniscate-world/Neural/issues
- Discussion forum: https://github.com/Lemniscate-world/Neural/discussions
- Email: Lemniscate_zero@proton.me

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
