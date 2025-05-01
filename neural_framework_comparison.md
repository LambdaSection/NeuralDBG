# Neural DSL vs. Popular ML Frameworks

## Code Comparison: MNIST Classifier

| Framework | Code Example | Lines of Code |
|-----------|--------------|---------------|
| **Neural DSL** | ```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    Conv2D(32, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(128, activation="relu")
    Dropout(0.5)
    Output(10, activation="softmax")
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  train {
    epochs: 15
    batch_size: 64
  }
}
``` | 15 |
| **TensorFlow** | ```python
import tensorflow as tf
from tensorflow.keras import layers

# Input layer
inputs = tf.keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Build model
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2
)
``` | 25 |
| **PyTorch** | ```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# Create model and optimizer
model = MNISTClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(15):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
``` | 35 |
| **ONNX Export** | ```python
# Neural DSL generates ONNX with one command
neural export mnist.neural --format onnx

# TensorFlow to ONNX
import tf2onnx
onnx_model, _ = tf2onnx.convert.from_keras(model)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# PyTorch to ONNX
torch.onnx.export(model, dummy_input, "model.onnx",
                 input_names=["input"],
                 output_names=["output"],
                 dynamic_axes={"input": {0: "batch_size"},
                              "output": {0: "batch_size"}})
``` | Multiple steps |

## Feature Comparison

| Feature | Neural DSL | TensorFlow | PyTorch | ONNX |
|---------|------------|------------|---------|------|
| **Syntax Complexity** | Low | Medium | Medium | N/A |
| **Framework Switching** | One-line flag | Rewrite code | Rewrite code | N/A |
| **Shape Validation** | Automatic | Runtime | Runtime | N/A |
| **HPO Support** | Built-in | External libs | External libs | N/A |
| **Debugging Tools** | NeuralDbg | TensorBoard | TensorBoard/PyTorch Profiler | Netron |
| **Code Generation** | Multiple backends | Single | Single | Runtime only |
| **Learning Curve** | Gentle | Steep | Steep | N/A |

## Hyperparameter Optimization Comparison

| Framework | HPO Code Example |
|-----------|------------------|
| **Neural DSL** | ```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    Dense(HPO(choice(128, 256)))
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
    Output(10, "softmax")
  optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
}
``` |
| **TensorFlow + Keras Tuner** | ```python
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))
    model.add(keras.layers.Dense(
        hp.Choice('units', [128, 256]),
        activation='relu'))
    model.add(keras.layers.Dropout(
        hp.Float('dropout', 0.3, 0.7, step=0.1)))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='my_dir',
    project_name='mnist')

tuner.search(x_train, y_train, epochs=10, validation_split=0.2)
``` |
| **PyTorch + Optuna** | ```python
def objective(trial):
    # Define hyperparameters
    units = trial.suggest_categorical('units', [128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7, step=0.1)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    # Create model with hyperparameters
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(units, 10),
        nn.Softmax(dim=1)
    )
    
    # Training code...
    
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
``` |

## Why Choose Neural DSL?

1. **Simplicity**: Write less code with a clean, declarative syntax
2. **Framework Agnostic**: Generate TensorFlow, PyTorch, or ONNX from the same code
3. **Built-in Debugging**: Powerful visualization and debugging tools included
4. **Automatic Shape Validation**: Catch errors before runtime
5. **Integrated HPO**: Optimize hyperparameters without external libraries
6. **Unified Training Config**: Configure all aspects in one place
