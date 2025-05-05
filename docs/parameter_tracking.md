# Parameter Tracking in Neural

Neural provides comprehensive parameter tracking capabilities to help you monitor, analyze, and optimize your neural network models. This guide explains how to use these features effectively.

## Table of Contents

1. [Introduction](#introduction)
2. [Experiment Tracking](#experiment-tracking)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
4. [Integration with External Tools](#integration-with-external-tools)
5. [CLI Commands](#cli-commands)
6. [Python API](#python-api)
7. [Best Practices](#best-practices)

## Introduction

Parameter tracking in Neural serves several important purposes:

- **Experiment Management**: Keep track of all your experiments, including hyperparameters, metrics, and artifacts
- **Hyperparameter Optimization**: Automatically find the best hyperparameters for your models
- **Performance Analysis**: Analyze model performance across different configurations
- **Reproducibility**: Ensure experiments can be reproduced exactly
- **Collaboration**: Share experiment results with team members

## Experiment Tracking

Neural's experiment tracking system allows you to:

- Track hyperparameters, metrics, and artifacts
- Visualize training progress
- Compare multiple experiments
- Analyze parameter importance

### Basic Usage

```bash
# Initialize experiment tracking
neural track init my_experiment

# Log hyperparameters
neural track log --hyperparameters '{"learning_rate": 0.001, "batch_size": 32}'

# Log metrics
neural track log --metrics '{"loss": 0.5, "accuracy": 0.85}' --step 1

# Log artifacts
neural track log --artifact model.h5 --framework tensorflow

# Show experiment details
neural track show <experiment_id>

# Plot metrics
neural track plot <experiment_id>

# Compare experiments
neural track compare <experiment_id1> <experiment_id2>
```

### Experiment Directory Structure

When you initialize an experiment, Neural creates a directory structure like this:

```
neural_experiments/
└── my_experiment_<id>/
    ├── metadata.json         # Experiment metadata
    ├── hyperparameters.json  # Hyperparameters
    ├── metrics.json          # Metrics history
    ├── artifacts.json        # Artifact metadata
    ├── summary.json          # Experiment summary
    ├── artifacts/            # Stored artifacts
    └── plots/                # Generated plots
```

## Hyperparameter Optimization

Neural provides built-in hyperparameter optimization (HPO) capabilities using various search strategies:

- **Bayesian Optimization**: Efficient search using Gaussian Processes
- **Evolutionary Algorithms**: Population-based search using genetic algorithms
- **Population-Based Training (PBT)**: Adaptive search that evolves hyperparameters during training

### HPO in Neural DSL

You can define hyperparameter search spaces directly in your Neural DSL code:

```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(HPO(choice(128, 256)), activation="relu")
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
    Output(10, activation="softmax")

  optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
  loss: "sparse_categorical_crossentropy"

  train {
    epochs: 10
    batch_size: 32
    validation_split: 0.2
    search_method: "bayesian"  # Use Bayesian optimization
  }
}
```

### Running HPO

```bash
# Compile and run HPO
neural compile model.neural --backend tensorflow --hpo --dataset MNIST

# Run HPO with more trials
neural run model.neural --hpo --dataset MNIST
```

### Analyzing HPO Results

After running HPO, you can analyze the results:

```bash
# Show parameter importance
neural track plot <experiment_id> --metrics parameter_importance

# Compare top trials
neural track compare <experiment_id1> <experiment_id2>
```

## Integration with External Tools

Neural can integrate with popular experiment tracking tools:

### MLflow

```bash
# Initialize experiment with MLflow integration
neural track init my_experiment --integration mlflow --tracking-uri http://localhost:5000

# Log metrics (will be sent to both Neural and MLflow)
neural track log --metrics '{"loss": 0.5, "accuracy": 0.85}' --step 1
```

### Weights & Biases (W&B)

```bash
# Initialize experiment with W&B integration
neural track init my_experiment --integration wandb --project-name neural-experiments

# Log metrics (will be sent to both Neural and W&B)
neural track log --metrics '{"loss": 0.5, "accuracy": 0.85}' --step 1
```

### TensorBoard

```bash
# Initialize experiment with TensorBoard integration
neural track init my_experiment --integration tensorboard --log-dir runs/neural

# Log metrics (will be sent to both Neural and TensorBoard)
neural track log --metrics '{"loss": 0.5, "accuracy": 0.85}' --step 1
```

## CLI Commands

Neural provides a comprehensive set of CLI commands for parameter tracking:

### Experiment Tracking

```bash
# Initialize experiment tracking
neural track init [experiment_name] [options]

# Log data to an experiment
neural track log [options]

# List all experiments
neural track list [options]

# Show experiment details
neural track show <experiment_id> [options]

# Plot experiment metrics
neural track plot <experiment_id> [options]

# Compare multiple experiments
neural track compare <experiment_id1> <experiment_id2> [options]
```

### Hyperparameter Optimization

```bash
# Compile with HPO
neural compile <file> --hpo [options]

# Run with HPO
neural run <file> --hpo [options]
```

## Python API

You can also use Neural's parameter tracking capabilities programmatically:

```python
from neural.tracking import ExperimentTracker, ExperimentManager
from neural.tracking.integrations import create_integration

# Create an experiment
manager = ExperimentManager()
experiment = manager.create_experiment("my_experiment")

# Log hyperparameters
experiment.log_hyperparameters({
    "learning_rate": 0.001,
    "batch_size": 32
})

# Log metrics
experiment.log_metrics({
    "loss": 0.5,
    "accuracy": 0.85
}, step=1)

# Log artifacts
experiment.log_artifact("model.h5")
experiment.log_model("model.h5", framework="tensorflow")

# Create a figure and log it
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
experiment.log_figure(fig, "my_plot")

# Integrate with external tools
mlflow_integration = create_integration("mlflow", experiment_name="my_experiment")
mlflow_integration.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=1)
```

## Best Practices

### Organizing Experiments

- Use meaningful experiment names
- Group related experiments together
- Add tags to experiments for easier filtering

### Tracking Hyperparameters

- Track all hyperparameters, even those not being optimized
- Include environment information (hardware, software versions)
- Log dataset information and preprocessing steps

### Tracking Metrics

- Log metrics at regular intervals
- Include validation metrics in addition to training metrics
- Track resource usage (memory, GPU utilization)

### Analyzing Results

- Compare experiments with different hyperparameters
- Analyze parameter importance to understand which parameters matter most
- Use parallel coordinates plots to visualize the hyperparameter space

### Reproducibility

- Set random seeds for all sources of randomness
- Log all dependencies and their versions
- Save model checkpoints at regular intervals

## Advanced Topics

### Custom Search Strategies

You can implement custom search strategies for HPO:

```python
from neural.hpo.strategies import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(name="my_custom_strategy")
        # Initialize your strategy

    def suggest_parameters(self, trial, search_space):
        # Implement your parameter suggestion logic
        return params
```

### Custom Metrics Collection

You can implement custom metrics collection:

```python
from neural.metrics.metrics_collector import MetricsCollector

class MyCustomMetricsCollector(MetricsCollector):
    def __init__(self, model_data, trace_data, backend='tensorflow'):
        super().__init__(model_data, trace_data, backend)
        # Initialize your collector

    def collect_custom_metrics(self, model, data):
        # Implement your metrics collection logic
        return metrics
```

### Distributed HPO

Neural supports distributed HPO for faster optimization:

```python
from neural.hpo.hpo import HPOptimizer

optimizer = HPOptimizer(
    strategy="bayesian",
    n_trials=100,
    parallel_trials=4,  # Run 4 trials in parallel
    timeout=3600        # 1 hour timeout
)

best_params = optimizer.optimize(
    neural_code,
    dataset_name="MNIST",
    backend="tensorflow"
)
```
