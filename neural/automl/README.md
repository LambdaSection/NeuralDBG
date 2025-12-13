# Neural AutoML - Automated Machine Learning and Neural Architecture Search

## Overview

The Neural AutoML module provides comprehensive automated machine learning capabilities including:

- **Multiple Search Strategies**: Grid Search, Random Search, Bayesian Optimization, Evolutionary Algorithms
- **Neural Architecture Search (NAS)**: Architecture space definition with DSL support
- **Parallel Execution**: Support for Ray and Dask distributed computing
- **Early Stopping**: Multiple pruning strategies (Median, Hyperband, ASHA)
- **HPO Integration**: Seamless integration with existing hyperparameter optimization module

## Features

### Search Strategies

1. **Grid Search**: Exhaustive search over discrete parameter space
2. **Random Search**: Random sampling from the search space
3. **Bayesian Optimization**: Gaussian Process-based optimization with acquisition functions (EI, UCB)
4. **Evolutionary Algorithms**: Genetic algorithm-based search with crossover and mutation
5. **Regularized Evolution**: Evolution with aging mechanism for better exploration

### Early Stopping Strategies

1. **Median Pruner**: Prune trials below median performance
2. **Hyperband**: Resource allocation with successive halving
3. **ASHA**: Asynchronous Successive Halving Algorithm
4. **Threshold Pruner**: Simple threshold-based pruning
5. **Percentile Pruner**: Prune below a percentile threshold
6. **Patient Pruner**: Stop trials with no improvement

### Execution Backends

1. **Sequential**: Single-threaded execution (default)
2. **Thread Pool**: Multi-threaded parallel execution
3. **Process Pool**: Multi-process parallel execution
4. **Ray**: Distributed execution with Ray
5. **Dask**: Distributed execution with Dask

## Installation

```bash
# Install with AutoML dependencies
pip install -e ".[hpo]"  # Core HPO dependencies

# For distributed execution
pip install ray>=2.0.0  # For Ray support
pip install dask[distributed]>=2023.0.0  # For Dask support
```

## Quick Start

### Basic AutoML Search

```python
from neural.automl import AutoMLEngine, ArchitectureSpace

# Define architecture space from DSL
dsl_config = """
network AutoMLNet {
    input: (28, 28, 1)
    
    Dense(units: hpo(categorical: [64, 128, 256])) -> relu
    Dropout(rate: hpo(range: [0.3, 0.7, step=0.1]))
    Dense(units: 10) -> softmax
    
    optimizer: adam(learning_rate: hpo(log_range: [1e-4, 1e-2]))
    training: {
        batch_size: hpo(categorical: [16, 32, 64])
    }
}
"""

# Create architecture space
space = ArchitectureSpace.from_dsl(dsl_config)

# Initialize AutoML engine
engine = AutoMLEngine(
    search_strategy='bayesian',
    early_stopping='median',
    executor_type='sequential',
    backend='pytorch',
    device='auto'
)

# Run search
results = engine.search(
    architecture_space=space,
    train_data=train_loader,
    val_data=val_loader,
    max_trials=50,
    max_epochs_per_trial=10
)

print(f"Best accuracy: {results['best_metrics']['val_acc']['max']:.4f}")
print(f"Best architecture: {results['best_architecture']}")
```

### NAS with Custom Operations

```python
from neural.automl import AutoMLEngine, ArchitectureSpace
from neural.automl.nas_operations import get_nas_primitives, create_nas_cell

# Get standard NAS operations
nas_ops = get_nas_primitives()

# Create a NAS cell
cell = create_nas_cell(
    operations=nas_ops,
    num_nodes=4,
    in_channels=64,
    out_channels=64
)

# Define custom architecture space
space = ArchitectureSpace()
space.input_shape = (32, 32, 3)
space.add_layer_choice('cell_1', [cell])
space.add_fixed_layer({'type': 'GlobalAveragePooling2D', 'params': {}})
space.add_fixed_layer({'type': 'Dense', 'params': {'units': 10}})

# Search
engine = AutoMLEngine(
    search_strategy='evolutionary',
    early_stopping='asha',
    backend='pytorch'
)

results = engine.search(
    architecture_space=space,
    train_data=train_loader,
    val_data=val_loader,
    max_trials=100
)
```

### Distributed Execution with Ray

```python
from neural.automl import AutoMLEngine, ArchitectureSpace

space = ArchitectureSpace.from_dsl(dsl_config)

# Use Ray for distributed execution
engine = AutoMLEngine(
    search_strategy='random',
    early_stopping='hyperband',
    executor_type='ray',
    max_workers=4,  # Number of parallel workers
    backend='pytorch'
)

results = engine.search(
    architecture_space=space,
    train_data=train_loader,
    val_data=val_loader,
    max_trials=100
)
```

### Integration with HPO Module

```python
from neural.automl import AutoMLEngine

engine = AutoMLEngine(backend='pytorch', device='cpu')

# Use HPO-integrated search
results = engine.search_with_hpo(
    dsl_config=dsl_config,
    train_data=train_loader,
    val_data=val_loader,
    n_trials=50,
    dataset_name='MNIST'
)
```

## Architecture Space DSL

### HPO Syntax

```python
# Categorical choice
Dense(units: hpo(categorical: [64, 128, 256]))

# Range (linear)
Dropout(rate: hpo(range: [0.1, 0.9, step=0.1]))

# Log range
optimizer: adam(learning_rate: hpo(log_range: [1e-5, 1e-2]))

# Training config
training: {
    batch_size: hpo(categorical: [16, 32, 64]),
    epochs: 50
}
```

### NAS Syntax

```python
# Layer choices
conv_layer = LayerChoice['Conv2D(filters=32, kernel_size=3)',
                         'Conv2D(filters=64, kernel_size=5)',
                         'SeparableConv2D(filters=32, kernel_size=3)']

# Parameter search
Dense(units=search_range(64, 512))
Conv2D(filters=search_choice([32, 64, 128]))
```

## Search Strategies Configuration

### Bayesian Optimization

```python
engine = AutoMLEngine(
    search_strategy='bayesian',
    acquisition_function='ei',  # 'ei', 'ucb', or 'pi'
    n_initial_random=10  # Random trials before GP
)
```

### Evolutionary Algorithm

```python
engine = AutoMLEngine(
    search_strategy='evolutionary',
    population_size=20,
    mutation_rate=0.2,
    crossover_rate=0.5,
    tournament_size=3
)
```

### Regularized Evolution

```python
engine = AutoMLEngine(
    search_strategy='regularized_evolution',
    population_size=20,
    sample_size=10
)
```

## Early Stopping Configuration

### Median Pruner

```python
engine = AutoMLEngine(
    early_stopping='median',
    n_startup_trials=5,
    n_warmup_steps=5
)
```

### Hyperband

```python
engine = AutoMLEngine(
    early_stopping='hyperband',
    max_resource=100,
    reduction_factor=3
)
```

### ASHA

```python
engine = AutoMLEngine(
    early_stopping='asha',
    reduction_factor=4,
    min_resource=1
)
```

## Advanced Features

### Custom NAS Operations

```python
from neural.automl.nas_operations import NASOperation

class CustomOperation(NASOperation):
    def __init__(self):
        super().__init__('custom_op')
    
    def to_layer_config(self, in_channels, out_channels):
        return {
            'type': 'CustomLayer',
            'params': {
                'in_channels': in_channels,
                'out_channels': out_channels
            }
        }
    
    def get_parameter_count(self, in_channels, out_channels):
        return in_channels * out_channels
```

### Architecture Registry

```python
from neural.automl.utils import ArchitectureRegistry

registry = ArchitectureRegistry()

# Register architectures
registry.register(architecture, metrics)

# Get cached results
cached = registry.get(architecture)

# Get top performers
top_10 = registry.get_top_k(k=10, metric='accuracy')
```

### Performance Prediction

```python
from neural.automl.evaluation import PerformancePredictor

predictor = PerformancePredictor(predictor_type='learning_curve')

# Predict from partial training
predicted_acc = predictor.predict(
    architecture,
    partial_metrics={'val_acc': [0.5, 0.6, 0.65, 0.68]}
)
```

## Architecture Export

```python
from neural.automl.utils import export_architecture_to_dsl

# Export best architecture to DSL
dsl_str = export_architecture_to_dsl(results['best_architecture'])

# Save to file
with open('best_architecture.neural', 'w') as f:
    f.write(dsl_str)
```

## Utilities

### Architecture Comparison

```python
from neural.automl.utils import compare_architectures

similarity = compare_architectures(arch1, arch2)  # Returns 0-1
```

### Training Time Estimation

```python
from neural.automl.utils import estimate_training_time

estimated_time = estimate_training_time(
    architecture,
    dataset_size=50000,
    batch_size=32,
    epochs=10
)
```

### Model Size Estimation

```python
from neural.automl.nas_operations import estimate_model_size, compute_flops

num_params = estimate_model_size(architecture)
flops = compute_flops(architecture, input_shape=(224, 224, 3))
```

## Best Practices

1. **Start with Random Search**: Good baseline before Bayesian/Evolutionary
2. **Use Early Stopping**: Significantly reduces search time
3. **Distributed Execution**: Use Ray/Dask for large-scale searches
4. **Monitor Resources**: Track memory and compute usage
5. **Save Checkpoints**: Enable checkpoint saving for long searches
6. **Validate Architectures**: Always validate on separate test set

## Examples

See `examples/automl_example.py` for comprehensive examples including:
- Basic AutoML search
- NAS with custom operations
- Distributed search with Ray
- HPO integration
- Multi-objective optimization

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
engine = AutoMLEngine(...)
# Or use smaller architectures in search space
```

### Slow Search

```python
# Enable early stopping
engine = AutoMLEngine(early_stopping='median')
# Or use distributed execution
engine = AutoMLEngine(executor_type='ray', max_workers=4)
```

### Ray Connection Issues

```python
# Specify Ray address explicitly
engine = AutoMLEngine(
    executor_type='ray',
    address='ray://localhost:10001'
)
```

## References

- ENAS: Efficient Neural Architecture Search
- DARTS: Differentiable Architecture Search
- Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
- ASHA: A System for Massively Parallel Hyperparameter Tuning
- Population Based Training of Neural Networks
