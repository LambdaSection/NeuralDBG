# Hyperparameter Optimization (HPO) Guide

This guide provides a comprehensive overview of hyperparameter optimization (HPO) in Neural DSL, including the latest features introduced in v0.2.7.

## Introduction to HPO

Hyperparameter optimization is the process of automatically finding the best hyperparameters for your neural network. Neural DSL provides a powerful and flexible HPO system that works seamlessly with both PyTorch and TensorFlow backends.

## HPO Syntax

Neural DSL uses a simple and intuitive syntax for HPO:

```yaml
HPO(choice(64, 128, 256))       # Choose from discrete values
HPO(range(0.1, 0.5, step=0.1))  # Linear range with step size
HPO(log_range(1e-4, 1e-2))      # Log-scale range (good for learning rates)
```

## Supported HPO Parameters

Neural DSL supports HPO for a wide range of parameters:

| Parameter | HPO Type | Example | Since Version |
|-----------|----------|---------|---------------|
| Dense units | `choice` | `Dense(HPO(choice(64, 128, 256)))` | v0.2.5 |
| Dropout rate | `range` | `Dropout(HPO(range(0.3, 0.7, step=0.1)))` | v0.2.5 |
| Learning rate | `log_range` | `Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))` | v0.2.5 |
| Conv2D filters | `choice` | `Conv2D(filters=HPO(choice(32, 64)))` | v0.2.6 |
| Conv2D kernel_size | `choice` | `Conv2D(kernel_size=HPO(choice((3,3), (5,5))))` | v0.2.7 |
| Padding | `choice` | `Conv2D(padding=HPO(choice("same", "valid")))` | v0.2.7 |
| Decay steps | `choice` | `ExponentialDecay(0.1, HPO(choice(500, 1000)), 0.96)` | v0.2.7 |

## Basic HPO Example

Here's a simple example of using HPO in a neural network:

```yaml
network SimpleHPOExample {
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
    search_method: "random"  # Use random search
  }
}
```

## Advanced HPO Example (v0.2.7+)

Neural DSL v0.2.7 introduces enhanced HPO support for Conv2D layers and learning rate schedules. Here's an advanced example that demonstrates these features:

```yaml
network AdvancedHPOExample {
  input: (28, 28, 1)
  layers:
    # Conv2D with HPO for filters, kernel_size, and padding
    Conv2D(
      filters=HPO(choice(32, 64)),
      kernel_size=HPO(choice((3,3), (5,5))),
      padding=HPO(choice("same", "valid")),
      activation="relu"
    )
    MaxPooling2D(pool_size=(2,2))

    # Another conv block with HPO
    Conv2D(
      filters=HPO(choice(64, 128)),
      kernel_size=HPO(choice((3,3), (5,5))),
      padding="same",
      activation="relu"
    )
    MaxPooling2D(pool_size=(2,2))

    # Flatten and dense layers
    Flatten()
    Dense(HPO(choice(128, 256, 512)), activation="relu")
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
    Output(10, "softmax")

  # Advanced optimizer configuration with HPO
  optimizer: Adam(
    learning_rate=ExponentialDecay(
      HPO(log_range(1e-3, 1e-1)),       # Initial learning rate
      HPO(choice(500, 1000, 2000)),      # Variable decay steps
      HPO(range(0.9, 0.99, step=0.01))   # Decay rate
    )
  )

  loss: "sparse_categorical_crossentropy"

  # Training configuration with HPO
  train {
    epochs: 20
    batch_size: HPO(choice(32, 64, 128))
    validation_split: 0.2
    search_method: "bayesian"  # Use Bayesian optimization
  }
}
```

## Search Methods

Neural DSL supports different search methods for HPO:

- **Random Search**: Simple and effective for many problems
  ```yaml
  search_method: "random"
  ```

- **Bayesian Optimization**: More efficient for expensive evaluations
  ```yaml
  search_method: "bayesian"
  ```

- **Grid Search**: Exhaustive search over all combinations
  ```yaml
  search_method: "grid"
  ```

## Running HPO

To run HPO, use the `--hpo` flag with the `compile` or `run` command:

```bash
neural compile my_model.neural --backend tensorflow --hpo
neural run my_model.neural --backend pytorch --hpo
```

## Visualizing HPO Results

After running HPO, you can visualize the results using the `visualize` command:

```bash
neural visualize hpo_results.json --format html
```

This will generate an interactive visualization of the HPO results, showing the relationship between hyperparameters and model performance.

## Best Practices

- **Use `log_range` for learning rates**: Learning rates often work better on a logarithmic scale
- **Start with a small search space**: Begin with a limited set of hyperparameters and gradually expand
- **Use Bayesian optimization for expensive evaluations**: If each model evaluation takes a long time, Bayesian optimization is more efficient
- **Consider the relationship between hyperparameters**: Some hyperparameters interact with each other (e.g., learning rate and batch size)

## Advanced Topics

### Custom Search Spaces

You can define custom search spaces for more complex hyperparameters:

```yaml
# Custom search space for a complex hyperparameter
HPO(custom({
  "option1": {"value1": 10, "value2": 20},
  "option2": {"value1": 30, "value2": 40}
}))
```

### Early Stopping

You can use early stopping to terminate unpromising trials early:

```yaml
train {
  early_stopping: true
  patience: 5  # Number of epochs with no improvement
  min_delta: 0.001  # Minimum change to qualify as improvement
}
```

### Parallel Trials

You can run multiple trials in parallel to speed up the search:

```bash
neural run my_model.neural --hpo --parallel 4  # Run 4 trials in parallel
```

## Conclusion

Hyperparameter optimization is a powerful technique for improving neural network performance. Neural DSL provides a flexible and intuitive system for HPO that works seamlessly with both PyTorch and TensorFlow backends. With the enhanced HPO support in v0.2.7, you can now optimize even more aspects of your neural network architecture.

For more information, see the [Neural DSL Documentation](https://github.com/Lemniscate-world/Neural/blob/main/docs/dsl.md#enhanced-hpo-support-v027).
