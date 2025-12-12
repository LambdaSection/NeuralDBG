"""
Benchmark model definitions in Neural DSL and equivalent raw TF/PyTorch implementations.
"""

def get_benchmark_models():
    """Returns dictionary of benchmark models in different formats."""
    return {
        'simple_mlp': {
            'neural_dsl': """network SimpleMLP {
    input: (28, 28, 1)
    layers:
        Flatten()
        Dense(units=128, activation="relu")
        Dropout(rate=0.2)
        Output(units=10, activation="softmax")
    
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
    
    train {
        epochs: 5
        batch_size: 128
        validation_split: 0.2
    }
}""",
            'tensorflow': """import tensorflow as tf
from tensorflow.keras import layers

def create_model():
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(units=10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model
""",
            'pytorch': """import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def create_model():
    model = SimpleMLP()
    return model
"""
        },
        'cnn': {
            'neural_dsl': """network CNN {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3,3), activation="relu")
        MaxPooling2D(pool_size=(2,2))
        Conv2D(filters=64, kernel_size=(3,3), activation="relu")
        MaxPooling2D(pool_size=(2,2))
        Flatten()
        Dense(units=128, activation="relu")
        Dropout(rate=0.5)
        Output(units=10, activation="softmax")
    
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
    
    train {
        epochs: 5
        batch_size: 128
        validation_split: 0.2
    }
}""",
            'tensorflow': """import tensorflow as tf
from tensorflow.keras import layers

def create_model():
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(units=10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model
""",
            'pytorch': """import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def create_model():
    model = CNN()
    return model
"""
        },
        'deep_mlp': {
            'neural_dsl': """network DeepMLP {
    input: (28, 28, 1)
    layers:
        Flatten()
        Dense(units=512, activation="relu")
        Dropout(rate=0.3)
        Dense(units=256, activation="relu")
        Dropout(rate=0.3)
        Dense(units=128, activation="relu")
        Dropout(rate=0.3)
        Output(units=10, activation="softmax")
    
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
    
    train {
        epochs: 5
        batch_size: 128
        validation_split: 0.2
    }
}""",
            'tensorflow': """import tensorflow as tf
from tensorflow.keras import layers

def create_model():
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs
    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.Dense(units=256, activation='relu')(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.Dense(units=10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model
""",
            'pytorch': """import torch
import torch.nn as nn
import torch.optim as optim

class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

def create_model():
    model = DeepMLP()
    return model
"""
        }
    }
