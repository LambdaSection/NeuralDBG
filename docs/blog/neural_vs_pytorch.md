# Neural DSL vs PyTorch: Declarative vs Imperative Neural Network Definition

When building neural networks, developers have traditionally relied on imperative frameworks like PyTorch, where you manually construct models using classes and methods. But what if there was a more concise, declarative way to define networks? Enter Neural DSL - a domain-specific language that lets you specify neural architectures in a clean, readable format.

## The Traditional PyTorch Approach

In PyTorch, defining a neural network requires creating a class that inherits from `nn.Module` and implementing the `forward` method:

```python
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

This approach gives you full control but requires boilerplate code and manual shape calculations.

## The Neural DSL Alternative

With Neural DSL, the same network becomes a clean, declarative specification:

```
network MNISTClassifier {
    input: (None, 28, 28, 1)
    layers:
      Conv2D(filters=32, kernel_size=(3,3), activation="relu")
      MaxPooling2D(pool_size=(2,2))
      Flatten()
      Dense(units=128, activation="relu")
      Dropout(rate=0.5)
      Output(units=10, activation="softmax")

    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)

    train {
      epochs: 15
      batch_size: 64
      validation_split: 0.2
    }
}
```

## Key Differences

**Conciseness**: Neural DSL reduces ~25 lines of PyTorch code to ~15 lines of declarative syntax.

**Readability**: The network structure is immediately apparent - you can see the layer sequence at a glance.

**Framework Agnostic**: Neural DSL can compile to PyTorch, TensorFlow, or ONNX, while PyTorch code is framework-specific.

**Validation**: Neural DSL includes built-in shape propagation and parameter validation.

**Training Configuration**: Training hyperparameters are co-located with the model definition.

## When to Use Each

- **Use PyTorch** when you need fine-grained control, custom layers, or complex forward passes
- **Use Neural DSL** for rapid prototyping, standard architectures, and when readability matters

Neural DSL bridges the gap between high-level APIs like Keras and low-level control of PyTorch, offering the best of both worlds for many use cases.