"""
Test data fixtures and constants for Aquarium E2E tests.

Centralized test data to avoid duplication across test files.
"""
from __future__ import annotations


SIMPLE_DSL = """network SimpleModel {
    input: (10,)
    layers:
        Dense(64, activation="relu")
        Output(10, activation="softmax")
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}"""


MNIST_DSL = """network MNISTClassifier {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=3, activation="relu")
        MaxPooling2D(pool_size=2)
        Conv2D(filters=64, kernel_size=3, activation="relu")
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(128, activation="relu")
        Dropout(0.5)
        Output(10, activation="softmax")
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}"""


CIFAR10_DSL = """network CIFAR10Model {
    input: (32, 32, 3)
    layers:
        Conv2D(filters=32, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(filters=64, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(filters=128, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(256, activation="relu")
        Dropout(0.3)
        Dense(128, activation="relu")
        Dropout(0.2)
        Output(10, activation="softmax")
    optimizer: Adam(learning_rate=0.0001)
    loss: categorical_crossentropy
}"""


IMAGENET_DSL = """network ImageNetModel {
    input: (224, 224, 3)
    layers:
        Conv2D(filters=64, kernel_size=7, strides=2, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=3, strides=2)
        Conv2D(filters=128, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(filters=256, kernel_size=3, activation="relu")
        BatchNormalization()
        Conv2D(filters=256, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(filters=512, kernel_size=3, activation="relu")
        BatchNormalization()
        Conv2D(filters=512, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(4096, activation="relu")
        Dropout(0.5)
        Dense(4096, activation="relu")
        Dropout(0.5)
        Output(1000, activation="softmax")
    optimizer: SGD(learning_rate=0.01, momentum=0.9)
    loss: categorical_crossentropy
}"""


RNN_DSL = """network TextClassifier {
    input: (100, 128)
    layers:
        LSTM(units=128, return_sequences=True)
        Dropout(0.3)
        LSTM(units=64)
        Dropout(0.3)
        Dense(64, activation="relu")
        Output(5, activation="softmax")
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}"""


TRANSFORMER_DSL = """network TransformerModel {
    input: (512, 256)
    layers:
        MultiHeadAttention(num_heads=8, key_dim=64)
        Dropout(0.1)
        Dense(512, activation="relu")
        Dropout(0.1)
        Dense(256, activation="relu")
        Flatten()
        Output(10, activation="softmax")
    optimizer: Adam(learning_rate=0.0001)
    loss: categorical_crossentropy
}"""


AUTOENCODER_DSL = """network Autoencoder {
    input: (784,)
    layers:
        Dense(256, activation="relu")
        Dense(128, activation="relu")
        Dense(64, activation="relu")
        Dense(128, activation="relu")
        Dense(256, activation="relu")
        Output(784, activation="sigmoid")
    optimizer: Adam(learning_rate=0.001)
    loss: mse
}"""


INVALID_DSL_MISSING_FIELD = """network InvalidModel {
    input: (10,)
    layers:
        Dense(64)
}"""


INVALID_DSL_SYNTAX_ERROR = """network InvalidModel {
    input: (10,)
    layers:
        Dense(64, activation="relu"
        Output(10)
    optimizer: Adam()
    loss: mse
}"""


INVALID_DSL_UNKNOWN_LAYER = """network InvalidModel {
    input: (10,)
    layers:
        UnknownLayer(param=value)
        Output(10)
    optimizer: Adam()
    loss: mse
}"""


BACKEND_CONFIGS = {
    "tensorflow": {
        "name": "TensorFlow",
        "value": "tensorflow",
        "extensions": [".pb", ".h5", ".keras"]
    },
    "pytorch": {
        "name": "PyTorch",
        "value": "pytorch",
        "extensions": [".pt", ".pth"]
    },
    "onnx": {
        "name": "ONNX",
        "value": "onnx",
        "extensions": [".onnx"]
    }
}


DATASET_CONFIGS = {
    "MNIST": {
        "name": "MNIST",
        "value": "MNIST",
        "input_shape": (28, 28, 1),
        "num_classes": 10
    },
    "CIFAR10": {
        "name": "CIFAR10",
        "value": "CIFAR10",
        "input_shape": (32, 32, 3),
        "num_classes": 10
    },
    "CIFAR100": {
        "name": "CIFAR100",
        "value": "CIFAR100",
        "input_shape": (32, 32, 3),
        "num_classes": 100
    },
    "ImageNet": {
        "name": "ImageNet",
        "value": "ImageNet",
        "input_shape": (224, 224, 3),
        "num_classes": 1000
    }
}


TRAINING_CONFIGS = {
    "quick": {
        "epochs": 1,
        "batch_size": 32,
        "validation_split": 0.1
    },
    "standard": {
        "epochs": 10,
        "batch_size": 32,
        "validation_split": 0.2
    },
    "thorough": {
        "epochs": 50,
        "batch_size": 64,
        "validation_split": 0.2
    },
    "production": {
        "epochs": 100,
        "batch_size": 128,
        "validation_split": 0.15
    }
}


OPTIMIZERS = [
    "Adam",
    "SGD",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adamax",
    "Nadam"
]


LOSS_FUNCTIONS = [
    "categorical_crossentropy",
    "binary_crossentropy",
    "sparse_categorical_crossentropy",
    "mse",
    "mae",
    "huber",
    "log_cosh"
]


ACTIVATION_FUNCTIONS = [
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "softplus",
    "softsign",
    "selu",
    "elu",
    "exponential"
]


EXAMPLE_MODELS = {
    "simple": SIMPLE_DSL,
    "mnist": MNIST_DSL,
    "cifar10": CIFAR10_DSL,
    "imagenet": IMAGENET_DSL,
    "rnn": RNN_DSL,
    "transformer": TRANSFORMER_DSL,
    "autoencoder": AUTOENCODER_DSL
}


INVALID_MODELS = {
    "missing_field": INVALID_DSL_MISSING_FIELD,
    "syntax_error": INVALID_DSL_SYNTAX_ERROR,
    "unknown_layer": INVALID_DSL_UNKNOWN_LAYER
}


UI_SELECTORS = {
    "editor": "#dsl-editor",
    "parse_button": "#parse-dsl-btn",
    "visualize_button": "#visualize-btn",
    "load_example_button": "#load-example-btn",
    "parse_status": "#parse-status",
    "model_info": "#model-info",
    "backend_select": "#runner-backend-select",
    "dataset_select": "#runner-dataset-select",
    "compile_button": "#runner-compile-btn",
    "run_button": "#runner-run-btn",
    "stop_button": "#runner-stop-btn",
    "export_button": "#runner-export-btn",
    "clear_button": "#runner-clear-btn",
    "console_output": "#runner-console-output",
    "status_badge": "#runner-status-badge",
    "epochs_input": "#runner-epochs",
    "batch_size_input": "#runner-batch-size",
    "val_split_input": "#runner-val-split",
    "export_modal": "#runner-export-modal",
    "export_filename": "#runner-export-filename",
    "export_location": "#runner-export-location",
    "export_confirm": "#runner-export-confirm",
    "export_cancel": "#runner-export-cancel"
}


EXPECTED_TIMEOUTS = {
    "page_load": 10000,
    "parse": 10000,
    "compile": 30000,
    "export": 5000,
    "tab_switch": 1000,
    "modal_open": 2000,
    "server_start": 30000
}


def get_dsl_by_name(name: str) -> str:
    """Get DSL example by name."""
    return EXAMPLE_MODELS.get(name.lower(), SIMPLE_DSL)


def get_invalid_dsl_by_type(error_type: str) -> str:
    """Get invalid DSL by error type."""
    return INVALID_MODELS.get(error_type, INVALID_DSL_SYNTAX_ERROR)


def get_backend_config(backend: str) -> dict:
    """Get backend configuration."""
    return BACKEND_CONFIGS.get(backend.lower(), BACKEND_CONFIGS["tensorflow"])


def get_dataset_config(dataset: str) -> dict:
    """Get dataset configuration."""
    return DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["MNIST"])


def get_training_config(config_name: str) -> dict:
    """Get training configuration preset."""
    return TRAINING_CONFIGS.get(config_name.lower(), TRAINING_CONFIGS["standard"])
