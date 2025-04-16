# Neural DSL v0.2.7: Enhanced HPO Support and Parser Improvements

*April 16, 2025*

We're excited to announce the release of Neural DSL v0.2.7, which brings significant improvements to hyperparameter optimization (HPO) support, particularly for convolutional layers and learning rate schedules.

## What's New in v0.2.7

### Enhanced HPO Support for Conv2D Layers

One of the most significant improvements in v0.2.7 is the enhanced HPO support for Conv2D layers. You can now optimize the `kernel_size` parameter using HPO, allowing for more flexible architecture search:

```yaml
# Conv2D with HPO for both filters and kernel_size
Conv2D(
  filters=HPO(choice(32, 64)),
  kernel_size=HPO(choice((3,3), (5,5))),
  padding=HPO(choice("same", "valid")),
  activation="relu"
)
```

This enhancement allows you to automatically search for the optimal kernel size configuration, which can significantly impact model performance, especially for computer vision tasks.

### Improved ExponentialDecay Parameter Structure

We've also improved the ExponentialDecay parameter structure to support more complex decay schedules with better parameter handling:

```yaml
# Enhanced ExponentialDecay with HPO for all parameters
optimizer: Adam(
  learning_rate=ExponentialDecay(
    HPO(log_range(1e-3, 1e-1)),       # Initial learning rate
    HPO(choice(500, 1000, 2000)),      # Variable decay steps
    HPO(range(0.9, 0.99, step=0.01))   # Decay rate
  )
)
```

This improvement allows for more flexible learning rate schedule optimization, which can lead to better convergence and performance.

### Extended Padding Options in Layers

We've extended HPO support to padding parameters, allowing you to optimize the padding strategy:

```yaml
# Conv2D with HPO for padding
Conv2D(
  filters=32,
  kernel_size=(3,3),
  padding=HPO(choice("same", "valid")),
  activation="relu"
)
```

This enhancement is particularly useful for computer vision tasks where the padding strategy can significantly impact the model's ability to capture features at the edges of images.

### Parser Improvements

We've made several improvements to the parser:

- Fixed metrics processing logic that was incorrectly placed in the exponential_decay method
- Improved HPO log_range parameter naming from low/high to min/max for consistency
- Enhanced HPO range handling with better step parameter defaults
- Removed redundant code in Conv2D kernel_size validation

These improvements make the Neural DSL more robust and easier to use, with more consistent parameter naming and better error handling.

## Getting Started with v0.2.7

You can install Neural DSL v0.2.7 using pip:

```bash
pip install neural-dsl==0.2.7
```

Or upgrade from a previous version:

```bash
pip install --upgrade neural-dsl
```

## Example: Advanced HPO Configuration

Here's a complete example that demonstrates the new HPO features in v0.2.7:

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

## What's Next?

We're continuously working to improve Neural DSL and make it more powerful and user-friendly. In upcoming releases, we plan to:

- Further enhance the NeuralPaper.ai integration for better model visualization and annotation
- Expand PyTorch support to match TensorFlow capabilities
- Improve documentation with more examples and tutorials
- Add support for more advanced HPO techniques

Stay tuned for more updates, and as always, we welcome your feedback and contributions!

## Get Involved

- GitHub: [https://github.com/Lemniscate-world/Neural](https://github.com/Lemniscate-world/Neural)
- Documentation: [https://github.com/Lemniscate-world/Neural/blob/main/docs/dsl.md](https://github.com/Lemniscate-world/Neural/blob/main/docs/dsl.md)
- Discord: [https://discord.gg/KFku4KvS](https://discord.gg/KFku4KvS)

Happy coding with Neural DSL!
