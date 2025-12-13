# Neural AutoML Quick Start Guide

## Installation

```bash
# Basic AutoML
pip install -e ".[automl]"

# With distributed execution
pip install -e ".[automl,distributed]"

# Full installation
pip install -e ".[full]"
```

## 5-Minute Quick Start

### 1. Basic Search

```python
from neural.automl import AutoMLEngine, ArchitectureSpace

# Define search space
dsl_config = """
network MyNet {
    input: (28, 28, 1)
    Dense(units: hpo(categorical: [64, 128, 256])) -> relu
    Dense(units: 10) -> softmax
    optimizer: adam(learning_rate: hpo(log_range: [1e-4, 1e-2]))
}
"""

# Create and run
space = ArchitectureSpace.from_dsl(dsl_config)
engine = AutoMLEngine(search_strategy='random', backend='pytorch')
results = engine.search(space, train_loader, val_loader, max_trials=20)

print(f"Best accuracy: {results['best_metrics']['val_acc']['max']:.4f}")
```

### 2. Distributed Search

```python
engine = AutoMLEngine(
    search_strategy='bayesian',
    executor_type='ray',
    max_workers=4
)
results = engine.search(space, train_loader, val_loader, max_trials=50)
```

### 3. With Early Stopping

```python
engine = AutoMLEngine(
    search_strategy='evolutionary',
    early_stopping='asha'
)
results = engine.search(space, train_loader, val_loader, max_trials=100)
```

## Common Patterns

### Grid Search

```python
engine = AutoMLEngine(search_strategy='grid')
```

### Bayesian Optimization

```python
engine = AutoMLEngine(
    search_strategy='bayesian',
    acquisition_function='ei',
    n_initial_random=10
)
```

### Evolutionary Search

```python
engine = AutoMLEngine(
    search_strategy='evolutionary',
    population_size=20,
    mutation_rate=0.2
)
```

## Early Stopping Options

```python
# Median pruner
engine = AutoMLEngine(early_stopping='median')

# Hyperband
engine = AutoMLEngine(early_stopping='hyperband', max_resource=100)

# ASHA
engine = AutoMLEngine(early_stopping='asha', reduction_factor=4)

# No early stopping
engine = AutoMLEngine(early_stopping=None)
```

## Execution Backends

```python
# Sequential (default)
engine = AutoMLEngine(executor_type='sequential')

# Thread pool
engine = AutoMLEngine(executor_type='thread', max_workers=4)

# Process pool
engine = AutoMLEngine(executor_type='process', max_workers=4)

# Ray (distributed)
engine = AutoMLEngine(executor_type='ray', max_workers=8)

# Dask (distributed)
engine = AutoMLEngine(executor_type='dask', max_workers=8)
```

## DSL Syntax

### HPO Parameters

```python
# Categorical choice
Dense(units: hpo(categorical: [64, 128, 256]))

# Linear range
Dropout(rate: hpo(range: [0.1, 0.9, step=0.1]))

# Log range
optimizer: adam(learning_rate: hpo(log_range: [1e-5, 1e-2]))

# Training config
training: {
    batch_size: hpo(categorical: [16, 32, 64])
}
```

### Custom Architecture Space

```python
space = ArchitectureSpace()
space.input_shape = (32, 32, 3)

space.add_layer_choice('conv', [
    {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': 3}},
    {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': 5}}
])

space.add_fixed_layer({'type': 'MaxPooling2D', 'params': {'pool_size': 2}})
```

## NAS Operations

```python
from neural.automl.nas_operations import get_nas_primitives, create_nas_cell

# Get standard operations
ops = get_nas_primitives()

# Create NAS cell
cell = create_nas_cell(ops, num_nodes=4, in_channels=64, out_channels=64)
```

## Results and Export

```python
# Get results
results = engine.search(...)

# Best architecture
best_arch = results['best_architecture']
best_metrics = results['best_metrics']

# Export to DSL
from neural.automl.utils import export_architecture_to_dsl
dsl_str = export_architecture_to_dsl(best_arch)

# Save
with open('best_model.neural', 'w') as f:
    f.write(dsl_str)
```

## Checkpointing

```python
# Automatic checkpointing
engine = AutoMLEngine(output_dir='./my_search')
results = engine.search(...)  # Checkpoints saved to ./my_search

# Load checkpoint
engine.load_checkpoint('./my_search/checkpoint_50.json')
```

## Advanced Features

### Performance Prediction

```python
from neural.automl.evaluation import PerformancePredictor

predictor = PerformancePredictor(predictor_type='learning_curve')
predicted_acc = predictor.predict(architecture, partial_metrics)
```

### Architecture Registry

```python
from neural.automl.utils import ArchitectureRegistry

registry = ArchitectureRegistry()
registry.register(architecture, metrics)
cached = registry.get(architecture)
top_10 = registry.get_top_k(k=10)
```

### Custom NAS Operations

```python
from neural.automl.nas_operations import NASOperation

class MyOp(NASOperation):
    def __init__(self):
        super().__init__('my_op')
    
    def to_layer_config(self, in_ch, out_ch):
        return {'type': 'MyLayer', 'params': {...}}
    
    def get_parameter_count(self, in_ch, out_ch):
        return in_ch * out_ch
```

## Tips and Best Practices

1. **Start Small**: Begin with 10-20 trials to validate setup
2. **Use Early Stopping**: Save 50-70% compute time
3. **Distributed for Large Searches**: Use Ray/Dask for 100+ trials
4. **Monitor Resources**: Check memory and GPU usage
5. **Save Checkpoints**: Enable for searches > 1 hour
6. **Validate Separately**: Always test on separate test set

## Troubleshooting

### Out of Memory
- Reduce batch size in training config
- Use smaller architectures in search space
- Enable early stopping

### Slow Search
- Use distributed executor (Ray/Dask)
- Enable aggressive early stopping
- Reduce epochs per trial

### Poor Results
- Increase number of trials
- Expand search space
- Use Bayesian or evolutionary search

## Examples

See:
- `examples/automl_example.py` - Comprehensive examples
- `neural/automl/README.md` - Full documentation
- `neural/automl/ARCHITECTURE.md` - Architecture details

## Quick Reference

| Feature | Command |
|---------|---------|
| Basic search | `AutoMLEngine().search(space, train, val, max_trials=20)` |
| Random search | `AutoMLEngine(search_strategy='random')` |
| Bayesian search | `AutoMLEngine(search_strategy='bayesian')` |
| Evolutionary | `AutoMLEngine(search_strategy='evolutionary')` |
| Early stopping | `AutoMLEngine(early_stopping='asha')` |
| Distributed | `AutoMLEngine(executor_type='ray', max_workers=4)` |
| HPO integration | `engine.search_with_hpo(dsl_config, train, val)` |

## Next Steps

1. Try the basic example above
2. Read `neural/automl/README.md` for detailed documentation
3. Run `examples/automl_example.py` for comprehensive examples
4. Explore `neural/automl/ARCHITECTURE.md` for implementation details
