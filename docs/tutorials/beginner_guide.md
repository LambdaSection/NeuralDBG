# Neural DSL Beginner's Guide: Build Your First Neural Network

Welcome! This guide is designed for people who are new to neural networks OR new to Neural DSL (or both). We'll start from the absolute basics and build up your skills step by step.

## What You'll Learn

By the end of this guide, you'll be able to:
- Understand what Neural DSL does and why it's useful
- Build a simple image classifier from scratch
- Visualize and debug your models
- Make your models better with experimentation

**Time Required:** 30-45 minutes

**Prerequisites:** 
- Python installed on your computer
- Basic familiarity with command line/terminal
- (Optional) Basic understanding of what neural networks are

## Part 1: Understanding Neural DSL (5 minutes)

### What Problem Does It Solve?

Imagine you're building a model to recognize handwritten digits. In traditional frameworks like TensorFlow or PyTorch, you'd write something like this:

```python
# Traditional approach - lots of boilerplate
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

With Neural DSL, you write this instead:

```neural
network DigitRecognizer {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3,3), "relu")
        MaxPooling2D((2,2))
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

**Benefits:**
- ✅ Less code, easier to read
- ✅ Framework-agnostic (compiles to TensorFlow, PyTorch, or ONNX)
- ✅ Built-in visualization and debugging
- ✅ Automatic shape validation before you run anything

## Part 2: Installation (5 minutes)

### Quick Install

```bash
# Basic installation (minimal dependencies)
pip install neural-dsl

# Or, install with all features (recommended for learning)
pip install neural-dsl[full]
```

**Troubleshooting:**
- If `pip` doesn't work, try `pip3`
- On Windows, you might need to run terminal as administrator
- If you get permission errors, add `--user`: `pip install --user neural-dsl`

### Verify Installation

```bash
neural --version
```

You should see version information. If you get "command not found", try:
```bash
python -m neural.cli --version
```

## Part 3: Your First Model (10 minutes)

Let's build a classifier for handwritten digits (MNIST dataset). Don't worry if you don't understand every part yet - we'll explain each piece.

### Step 1: Create Your Model File

Create a new file called `my_first_model.neural`:

```neural
# This is a comment - anything after # is ignored
# Let's build a digit classifier!

network DigitClassifier {
    # Input: 28x28 grayscale images
    # Format is (height, width, channels)
    input: (28, 28, 1)
    
    # Layers process the image step by step
    layers:
        # Convolutional layer: looks for patterns
        # 32 = number of patterns to find
        # (3,3) = pattern size
        # "relu" = activation function (helps model learn complex patterns)
        Conv2D(32, (3,3), "relu")
        
        # Pooling: reduces size, keeps important info
        # (2,2) = shrink by 2 in each dimension
        MaxPooling2D((2,2))
        
        # Another conv layer for more complex patterns
        Conv2D(64, (3,3), "relu")
        MaxPooling2D((2,2))
        
        # Flatten: convert 2D image to 1D list
        Flatten()
        
        # Dense: fully connected layer for decision making
        # 128 = number of neurons
        Dense(128, "relu")
        
        # Dropout: prevents overfitting (memorizing training data)
        Dropout(0.5)
        
        # Output: 10 neurons (one for each digit 0-9)
        # "softmax" = converts to probabilities
        Output(10, "softmax")
    
    # Loss function: measures how wrong predictions are
    loss: "sparse_categorical_crossentropy"
    
    # Optimizer: algorithm that improves the model
    # learning_rate: how fast to learn (0.001 is a good default)
    optimizer: Adam(learning_rate=0.001)
    
    # Training configuration
    train {
        epochs: 10           # How many times to see full dataset
        batch_size: 64       # Process 64 images at once
        validation_split: 0.2  # Use 20% of data for testing
    }
}
```

### Step 2: Visualize Your Model

Before running anything, let's see what our model looks like:

```bash
neural visualize my_first_model.neural --format png
```

This creates `my_first_model.png` showing your network architecture. Open it to see how layers connect.

**What to Look For:**
- Each box is a layer
- Arrows show data flow
- Numbers show tensor shapes (dimensions)

### Step 3: Check for Errors

Neural DSL validates your model before running it:

```bash
neural compile my_first_model.neural --dry-run
```

This checks:
- ✅ Syntax is correct
- ✅ Layer dimensions match up
- ✅ All parameters are valid

If there are errors, read the message carefully - it will tell you exactly what's wrong and where.

### Step 4: Generate Real Code

Now compile to actual Python code:

```bash
# Generate TensorFlow code
neural compile my_first_model.neural --backend tensorflow --output digit_classifier.py

# Or PyTorch code
neural compile my_first_model.neural --backend pytorch --output digit_classifier_torch.py
```

**Open the generated file** to see the Python code. This is real, working code you can run or modify.

## Part 4: Understanding Shapes (Important!)

The most common errors in neural networks are **shape mismatches**. Let's understand this:

### What Are Shapes?

A tensor shape describes the dimensions of data:
- `(28, 28, 1)` = 28 pixels high, 28 wide, 1 color channel (grayscale)
- `(32, 32, 3)` = 32x32 pixels, 3 channels (RGB color)
- `(128,)` = a flat list of 128 numbers

### How Layers Change Shapes

```neural
input: (28, 28, 1)           # Start: 28x28 image

Conv2D(32, (3,3))            # After: (26, 26, 32)
# Why? 3x3 filter reduces edge by 1 pixel each side
# 32 = number of filters (output channels)

MaxPooling2D((2,2))          # After: (13, 13, 32)
# Divides dimensions by 2

Flatten()                    # After: (5408,)
# Converts 13 * 13 * 32 = 5408

Dense(128)                   # After: (128,)
# Output size you specify
```

### Debug Shape Issues

If you get shape errors:

```bash
neural debug my_first_model.neural
```

This opens a dashboard showing exactly how shapes change through each layer.

## Part 5: Making It Better (10 minutes)

Now that you have a working model, let's improve it!

### Experiment 1: Add More Layers

Try adding another convolutional block:

```neural
layers:
    Conv2D(32, (3,3), "relu")
    MaxPooling2D((2,2))
    Conv2D(64, (3,3), "relu")
    MaxPooling2D((2,2))
    # ADD THIS:
    Conv2D(128, (3,3), "relu")
    MaxPooling2D((2,2))
    # ...rest of layers
```

**Re-visualize** to see how this changes the architecture:
```bash
neural visualize my_first_model.neural --format png
```

### Experiment 2: Try Batch Normalization

Batch normalization helps models train faster and better:

```neural
layers:
    Conv2D(32, (3,3), "relu")
    BatchNormalization()    # ADD THIS after each Conv2D
    MaxPooling2D((2,2))
    
    Conv2D(64, (3,3), "relu")
    BatchNormalization()    # ADD THIS
    MaxPooling2D((2,2))
    # ...
```

### Experiment 3: Adjust Hyperparameters

Try different learning rates:

```neural
# Faster learning (might be unstable)
optimizer: Adam(learning_rate=0.01)

# Slower learning (more stable but takes longer)
optimizer: Adam(learning_rate=0.0001)
```

**Pro Tip:** Learning rate is one of the most important hyperparameters. Too high = unstable, too low = too slow.

### Experiment 4: Automatic Hyperparameter Search

Let Neural DSL find good hyperparameters for you:

```neural
network DigitClassifier {
    input: (28, 28, 1)
    layers:
        # HPO() tells Neural to try different values
        Conv2D(HPO(choice(32, 64, 128)), (3,3), "relu")
        MaxPooling2D((2,2))
        Flatten()
        Dense(HPO(choice(128, 256, 512)), "relu")
        Dropout(HPO(range(0.3, 0.7, step=0.1)))
        Output(10, "softmax")
    
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=HPO(log_range(0.0001, 0.01)))
    
    train {
        epochs: 10
        batch_size: 64
        search_method: "bayesian"  # Smart search strategy
    }
}
```

Then compile with HPO enabled:

```bash
neural compile my_first_model.neural --backend tensorflow --hpo
```

This will try different combinations and find what works best.

## Part 6: Common Issues and Solutions (5 minutes)

### Issue: "Shape mismatch error"

**Problem:** Layers don't connect properly.

**Solution:**
1. Run `neural debug my_first_model.neural`
2. Look at the shape transformation diagram
3. Find where shapes don't match
4. Usually fixed by adding `Flatten()` before `Dense()` layers

### Issue: "Model trains but accuracy is terrible"

**Problem:** Architecture or hyperparameters need adjustment.

**Solutions:**
- Add more layers (deeper network)
- Add more neurons per layer (wider network)
- Adjust learning rate
- Try different optimizers (SGD, RMSprop, Adam)
- Add regularization (Dropout, BatchNormalization)

### Issue: "Training is very slow"

**Solutions:**
- Reduce model size (fewer layers/neurons)
- Increase batch size
- Use GPU if available: `execution { device: "cuda" }`
- Reduce number of epochs

### Issue: "Model overfits (good training accuracy, bad validation accuracy)"

**Solutions:**
- Add Dropout layers
- Use smaller model (fewer parameters)
- Get more training data
- Add data augmentation

## Part 7: What's Next?

### Practice Projects

Try building these models:

1. **Fashion MNIST**: Similar to digits, but classifying clothing items
   - Change input to `(28, 28, 1)` (same as digits)
   - Change output to `Output(10, "softmax")` (10 clothing types)

2. **CIFAR-10**: Classifying color images (airplanes, cars, birds, etc.)
   - Change input to `(32, 32, 3)` (color images)
   - Need deeper network (more Conv2D layers)

3. **Text Sentiment**: Classify movie reviews as positive/negative
   - Use `Embedding()` and `LSTM()` layers
   - See `examples/sentiment.neural` for inspiration

### Advanced Features

Once comfortable with basics, explore:

- **No-Code Interface**: Build models visually
  ```bash
  neural --no-code
  # Opens web interface on http://localhost:8051
  ```

- **Real-Time Debugging**: Watch your model train
  ```bash
  neural debug my_first_model.neural
  # Opens dashboard on http://localhost:8050
  ```

- **Cloud Deployment**: Run on Kaggle, Colab, or AWS
  ```bash
  neural cloud connect kaggle
  ```

### Learning Resources

- **Example Models**: See `examples/` directory for more models
- **Documentation**: Read `docs/dsl.md` for complete syntax reference
- **Notebooks**: Check `examples/notebooks/` for interactive tutorials
- **Community**: Ask questions on GitHub Discussions

## Quick Reference Card

### Basic Model Structure

```neural
network ModelName {
    input: (height, width, channels)
    layers:
        # Your layers here
        Conv2D(filters, (kernel_h, kernel_w), "activation")
        MaxPooling2D((pool_h, pool_w))
        Flatten()
        Dense(units, "activation")
        Output(num_classes, "softmax")
    loss: "loss_function"
    optimizer: OptimizerName(learning_rate=0.001)
    train {
        epochs: 10
        batch_size: 32
    }
}
```

### Common Layer Types

| Layer | Purpose | Example |
|-------|---------|---------|
| `Conv2D` | Find patterns in images | `Conv2D(32, (3,3), "relu")` |
| `MaxPooling2D` | Reduce size | `MaxPooling2D((2,2))` |
| `Flatten` | Convert 2D to 1D | `Flatten()` |
| `Dense` | Fully connected | `Dense(128, "relu")` |
| `Dropout` | Prevent overfitting | `Dropout(0.5)` |
| `BatchNormalization` | Normalize activations | `BatchNormalization()` |
| `LSTM` | Process sequences | `LSTM(64)` |
| `Embedding` | Convert text to numbers | `Embedding(10000, 128)` |

### Essential Commands

```bash
# Visualize architecture
neural visualize model.neural --format png

# Check for errors
neural compile model.neural --dry-run

# Generate code
neural compile model.neural --backend tensorflow

# Debug interactively
neural debug model.neural

# Hyperparameter optimization
neural compile model.neural --hpo
```

## Getting Help

**Got stuck?** That's normal when learning!

1. **Re-read** relevant sections of this guide
2. **Check examples** in the `examples/` directory
3. **Search** GitHub issues for similar problems
4. **Ask** on GitHub Discussions
5. **Open** an issue if you found a bug

**Remember:** Every expert was once a beginner. Take your time, experiment, and don't be afraid to make mistakes!

---

**Congratulations!** 🎉 You now know the fundamentals of Neural DSL. Start building, experimenting, and learning. The best way to improve is to build real projects!

Happy modeling! 🚀
