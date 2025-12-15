# Contributing to Neural DSL Benchmarks

Thank you for your interest in improving the Neural DSL benchmarking suite! This guide will help you contribute effectively.

## Ways to Contribute

### 1. Add New Framework Implementations

We're always looking to expand our framework comparisons. To add a new framework:

**Step 1: Create Implementation**

```python
# In framework_implementations.py

class MyFrameworkImplementation(FrameworkImplementation):
    def __init__(self):
        super().__init__("My Framework")
    
    def setup(self):
        # Define the code string that would be written
        self.code_content = """
        import myframework
        
        model = myframework.Model([
            myframework.layers.Conv2D(32, 3, activation='relu'),
            # ... rest of model definition
        ])
        """
    
    def build_model(self):
        # Actually build the model
        import myframework
        self.model = myframework.Model([...])
    
    def train(self, dataset: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        # Train the model and return metrics
        # Must return: training_time, accuracy, val_accuracy, val_loss, 
        #              training_loss, peak_memory_mb, error_rate
        pass
    
    def predict_single(self) -> Any:
        # Make a single prediction
        pass
    
    def _save_model(self, path: str):
        # Save model to disk
        pass
    
    def get_parameter_count(self) -> int:
        # Return total trainable parameters
        pass
```

**Step 2: Update Exports**

Add your implementation to `__init__.py`:

```python
from .framework_implementations import (
    # ... existing imports
    MyFrameworkImplementation,
)

__all__ = [
    # ... existing exports
    "MyFrameworkImplementation",
]
```

**Step 3: Update CLI**

Add to `run_benchmarks.py`:

```python
framework_map = {
    # ... existing frameworks
    "myframework": MyFrameworkImplementation,
}
```

**Step 4: Test**

```bash
python neural/benchmarks/run_benchmarks.py --frameworks neural myframework --epochs 2
```

**Step 5: Document**

Update `README.md` to mention the new framework.

### 2. Add New Metrics

To track additional performance indicators:

**Step 1: Extend MetricsCollector**

```python
# In metrics_collector.py

class EnhancedMetricsCollector(MetricsCollector):
    def collect_gpu_memory(self):
        """Collect GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 2)
        except ImportError:
            pass
        return 0.0
```

**Step 2: Update BenchmarkResult**

```python
# In benchmark_runner.py

@dataclass
class BenchmarkResult:
    # ... existing fields
    gpu_memory_mb: float = 0.0
    new_metric: float = 0.0
```

**Step 3: Collect in Runner**

```python
# In benchmark_runner.py, run_benchmark method

result = BenchmarkResult(
    # ... existing fields
    gpu_memory_mb=metrics.get("gpu_memory_mb", 0),
)
```

### 3. Add New Visualizations

To create new chart types:

**Step 1: Implement in BenchmarkVisualizer**

```python
# In visualization.py

def plot_my_custom_chart(
    self,
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
):
    """Create a custom visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Your plotting code here
    # ...
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig, ax
```

**Step 2: Add to generate_all_plots**

```python
def generate_all_plots(self, results_path: str, output_dir: Path):
    # ... existing plots
    
    self.plot_my_custom_chart(
        df,
        output_path=output_dir / "my_custom_chart.png"
    )
    print("  ✓ My custom chart")
```

### 4. Add New Benchmark Tasks

To expand beyond MNIST:

**Step 1: Update Configuration**

```yaml
# In benchmark_config.yaml

tasks:
  - name: CIFAR10_Classification
    description: "CNN for CIFAR-10 image classification"
    dataset: cifar10
    model_type: cnn
    epochs: 10
    batch_size: 128
    expected_accuracy: 0.75
    priority: high
```

**Step 2: Implement Dataset Loading**

```python
# In framework implementations

def train(self, dataset: str, epochs: int, batch_size: int):
    if dataset == "cifar10":
        # Load CIFAR-10
        from tensorflow.keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # ... process data
    elif dataset == "mnist":
        # ... existing MNIST code
```

**Step 3: Update All Implementations**

Make sure all framework implementations support the new dataset.

### 5. Improve Documentation

Documentation improvements are always welcome:

- Fix typos and grammar
- Add clarifying examples
- Improve code comments
- Add troubleshooting tips
- Translate to other languages

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/your-org/neural-dsl.git
cd neural-dsl
pip install -e ".[full]"
pip install -r neural/benchmarks/requirements.txt
```

### 2. Run Tests

```bash
# Quick test
python neural/benchmarks/quick_start.py

# Full test
python neural/benchmarks/run_benchmarks.py --frameworks neural keras --epochs 2
```

### 3. Make Changes

Edit the relevant files following the patterns above.

### 4. Test Your Changes

```bash
# Test specific component
python -c "from neural.benchmarks import MyNewClass; MyNewClass().test()"

# Test full benchmark
python neural/benchmarks/run_benchmarks.py --frameworks neural --epochs 1
```

## Code Style

### Python Style

Follow PEP 8 and the project's existing patterns:

- Use type hints
- Add docstrings to public functions
- Keep lines under 100 characters
- Use descriptive variable names

```python
def calculate_speedup(
    baseline_time: float,
    comparison_time: float,
) -> float:
    """
    Calculate speedup relative to baseline.
    
    Args:
        baseline_time: Reference time in seconds
        comparison_time: Time to compare in seconds
    
    Returns:
        Speedup factor (higher is faster)
    """
    if comparison_time <= 0:
        return 0.0
    return baseline_time / comparison_time
```

### Documentation Style

Use clear, concise language:

```markdown
## Good

Run benchmarks with custom parameters:
\`\`\`bash
python run_benchmarks.py --epochs 10
\`\`\`

## Less Good

You can run the benchmarks and specify different parameters 
if you want to by using the command line arguments like this:
\`\`\`bash
python run_benchmarks.py --epochs 10
\`\`\`
```

## Testing Guidelines

### What to Test

1. **New Framework Implementations**
   - Model builds successfully
   - Training completes without errors
   - Metrics are collected correctly
   - LOC counting is accurate

2. **New Metrics**
   - Values are reasonable
   - Doesn't slow down benchmarks
   - Works across different frameworks

3. **New Visualizations**
   - Charts render correctly
   - Data is accurate
   - Labels are clear
   - Colors are distinct

### How to Test

```bash
# Manual testing
python neural/benchmarks/example_benchmark.py --quick

# Check output
ls -l benchmark_results/
ls -l benchmark_reports/

# Verify charts
open benchmark_reports/neural_dsl_benchmark_*/index.html
```

## Submitting Changes

### 1. Create Branch

```bash
git checkout -b feature/my-new-framework
```

### 2. Make Changes

Edit files, add features, fix bugs.

### 3. Test Locally

```bash
python neural/benchmarks/run_benchmarks.py --frameworks neural --epochs 1
```

### 4. Commit

```bash
git add neural/benchmarks/framework_implementations.py
git commit -m "Add MyFramework implementation for benchmarking"
```

### 5. Push

```bash
git push origin feature/my-new-framework
```

### 6. Open Pull Request

- Describe what you changed and why
- Include benchmark results if applicable
- Reference any related issues

## PR Checklist

Before submitting a PR:

- [ ] Code follows project style
- [ ] All existing tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Benchmark results included (if applicable)
- [ ] No merge conflicts
- [ ] Commit messages are clear

## Questions?

- **GitHub Issues**: [Ask questions](https://github.com/your-org/neural-dsl/issues)
- **Discord**: [Join our community](#)
- **Email**: benchmarks@neural-dsl.org

## Recognition

All contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make Neural DSL benchmarks better! 🚀
