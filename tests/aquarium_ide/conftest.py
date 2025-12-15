"""
Pytest configuration and fixtures for Aquarium IDE integration tests.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock


@pytest.fixture
def aquarium_backend_url():
    """Backend server URL for Aquarium IDE."""
    return "http://localhost:8000"


@pytest.fixture
def aquarium_frontend_url():
    """Frontend server URL for Aquarium IDE."""
    return "http://localhost:3000"


@pytest.fixture
def sample_templates():
    """Sample quick start templates for testing."""
    return [
        {
            "id": "image-classification",
            "title": "Image Classification",
            "description": "CNN for classifying images into categories",
            "category": "Computer Vision",
            "icon": "🖼️",
            "difficulty": "beginner",
            "dsl_code": """network ImageClassifier {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation="relu")
        Dense(units=10, activation="softmax")
    loss: "categorical_crossentropy"
    optimizer: "Adam"
}"""
        },
        {
            "id": "text-classification",
            "title": "Text Classification",
            "description": "LSTM for sentiment analysis and text categorization",
            "category": "NLP",
            "icon": "📝",
            "difficulty": "beginner",
            "dsl_code": """network TextClassifier {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=128)
        LSTM(units=64)
        Dense(units=1, activation="sigmoid")
    loss: "binary_crossentropy"
    optimizer: "Adam"
}"""
        },
        {
            "id": "time-series",
            "title": "Time Series Forecasting",
            "description": "LSTM network for predicting future values",
            "category": "Time Series",
            "icon": "📈",
            "difficulty": "intermediate",
            "dsl_code": """network TimeSeriesForecaster {
    input: (None, 50, 1)
    layers:
        LSTM(units=128, return_sequences=true)
        Dropout(rate=0.2)
        LSTM(units=64)
        Dense(units=1)
    loss: "mse"
    optimizer: "Adam"
}"""
        }
    ]


@pytest.fixture
def sample_examples_metadata():
    """Sample example metadata for testing."""
    return [
        {
            "name": "MNIST CNN",
            "path": "examples/mnist_cnn.neural",
            "description": "Convolutional Neural Network for MNIST digit classification",
            "category": "Computer Vision",
            "tags": ["cnn", "classification", "mnist"],
            "complexity": "Beginner"
        },
        {
            "name": "LSTM Text Classifier",
            "path": "examples/lstm_text.neural",
            "description": "LSTM network for text classification and sentiment analysis",
            "category": "NLP",
            "tags": ["lstm", "nlp", "text", "classification"],
            "complexity": "Beginner"
        },
        {
            "name": "ResNet Image Classifier",
            "path": "examples/resnet.neural",
            "description": "Deep residual network for advanced image classification",
            "category": "Computer Vision",
            "tags": ["resnet", "cnn", "deep-learning"],
            "complexity": "Advanced"
        }
    ]


@pytest.fixture
def example_neural_files(tmp_path):
    """Create temporary .neural example files."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    
    examples = {
        "mnist_cnn.neural": """network MNISTClassifier {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        MaxPooling2D(pool_size=(2, 2))
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu")
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation="relu")
        Dropout(rate=0.5)
        Dense(units=10, activation="softmax")
    loss: "categorical_crossentropy"
    optimizer: "Adam"
}""",
        "lstm_text.neural": """network TextClassifier {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=128)
        LSTM(units=64, return_sequences=true)
        LSTM(units=64)
        Dense(units=64, activation="relu")
        Dropout(rate=0.5)
        Dense(units=1, activation="sigmoid")
    loss: "binary_crossentropy"
    optimizer: "Adam"
}""",
        "resnet.neural": """network ResNetClassifier {
    input: (None, 224, 224, 3)
    layers:
        Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation="relu")
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        Conv2D(filters=128, kernel_size=(3, 3), activation="relu")
        Conv2D(filters=128, kernel_size=(3, 3), activation="relu")
        GlobalAveragePooling2D()
        Dense(units=1000, activation="softmax")
    loss: "categorical_crossentropy"
    optimizer: "Adam"
}"""
    }
    
    for filename, content in examples.items():
        (examples_dir / filename).write_text(content)
    
    return examples_dir


@pytest.fixture
def sample_dsl_codes():
    """Common DSL code samples for testing."""
    return {
        "simple": """network SimpleNet {
    input: (28, 28, 1)
    layers:
        Flatten()
        Dense(128, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}""",
        "cnn": """network CNNModel {
    input: (32, 32, 3)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Conv2D(64, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(128, "relu")
        Dropout(0.5)
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}""",
        "lstm": """network LSTMModel {
    input: (None, 100)
    layers:
        Embedding(10000, 128)
        LSTM(64, return_sequences=true)
        LSTM(64)
        Dense(32, "relu")
        Dense(1, "sigmoid")
    loss: "binary_crossentropy"
    optimizer: "adam"
}""",
        "transformer": """network TransformerModel {
    input: (128, 512)
    layers:
        TransformerEncoder(num_heads=8, ff_dim=2048)
        GlobalAveragePooling1D()
        Dense(256, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}"""
    }


@pytest.fixture
def mock_fastapi_app():
    """Mock FastAPI app for testing."""
    app = MagicMock()
    app.title = "Neural DSL Backend Bridge"
    app.version = "0.3.0"
    return app


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing."""
    return {
        "health": {"status": "healthy"},
        "root": {
            "service": "Neural DSL Backend Bridge",
            "version": "0.3.0",
            "status": "running"
        },
        "examples_list": {
            "examples": [
                {
                    "name": "MNIST CNN",
                    "path": "examples/mnist_cnn.neural",
                    "description": "CNN for MNIST",
                    "category": "Computer Vision",
                    "tags": ["cnn", "mnist"],
                    "complexity": "Beginner"
                }
            ],
            "count": 1
        },
        "example_load": {
            "code": "network MNISTClassifier {}",
            "path": "examples/mnist_cnn.neural",
            "name": "mnist_cnn"
        }
    }


@pytest.fixture
def welcome_screen_state():
    """Welcome screen state for testing."""
    return {
        "visible": True,
        "active_tab": "quickstart",
        "show_on_startup": True,
        "tabs": ["quickstart", "examples", "docs", "videos"]
    }


@pytest.fixture
def editor_state():
    """Editor state for testing."""
    return {
        "code": "",
        "language": "neural-dsl",
        "theme": "dark",
        "fontSize": 14,
        "readOnly": False
    }


@pytest.fixture
def compilation_state():
    """Compilation state for testing."""
    return {
        "is_compiling": False,
        "success": None,
        "error": None,
        "model_data": None,
        "generated_code": None,
        "backend": "tensorflow"
    }


@pytest.fixture
def mock_callbacks():
    """Mock callback functions for testing."""
    return {
        "on_close": Mock(),
        "on_load_template": Mock(),
        "on_load_example": Mock(),
        "on_start_tutorial": Mock(),
        "on_compile": Mock(),
        "on_export": Mock()
    }


@pytest.fixture
def api_endpoints():
    """API endpoint URLs for testing."""
    base_url = "http://localhost:8000"
    return {
        "root": f"{base_url}/",
        "health": f"{base_url}/health",
        "parse": f"{base_url}/api/parse",
        "shape_propagation": f"{base_url}/api/shape-propagation",
        "generate_code": f"{base_url}/api/generate-code",
        "compile": f"{base_url}/api/compile",
        "examples_list": f"{base_url}/api/examples/list",
        "examples_load": f"{base_url}/api/examples/load",
        "jobs": f"{base_url}/api/jobs",
        "docs": f"{base_url}/api/docs"
    }


@pytest.fixture
def valid_categories():
    """Valid example and template categories."""
    return [
        "Computer Vision",
        "NLP",
        "Time Series",
        "Generative",
        "Unsupervised",
        "Reinforcement Learning",
        "General"
    ]


@pytest.fixture
def valid_complexity_levels():
    """Valid complexity/difficulty levels."""
    return ["Beginner", "Intermediate", "Advanced"]


@pytest.fixture
def valid_backends():
    """Valid backend targets."""
    return ["tensorflow", "pytorch", "onnx"]


@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "invalid_dsl": "this is not valid dsl code",
        "missing_file": "nonexistent.neural",
        "invalid_extension": "example.txt",
        "network_error": {"error": "Network timeout"},
        "parse_error": {"success": False, "error": "Parse failed"},
        "compilation_error": {"success": False, "error": "Compilation failed"}
    }


def pytest_configure(config):
    """Configure pytest for Aquarium IDE tests."""
    config.addinivalue_line(
        "markers",
        "aquarium: Aquarium IDE integration tests"
    )
    config.addinivalue_line(
        "markers",
        "welcome_screen: Welcome screen tests"
    )
    config.addinivalue_line(
        "markers",
        "templates: Template tests"
    )
    config.addinivalue_line(
        "markers",
        "examples: Example gallery tests"
    )
    config.addinivalue_line(
        "markers",
        "api: API endpoint tests"
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end workflow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on file and function names."""
    for item in items:
        if "aquarium_ide" in str(item.fspath):
            item.add_marker(pytest.mark.aquarium)
            item.add_marker(pytest.mark.integration)
        
        if "welcome" in item.name.lower():
            item.add_marker(pytest.mark.welcome_screen)
        
        if "template" in item.name.lower():
            item.add_marker(pytest.mark.templates)
        
        if "example" in item.name.lower():
            item.add_marker(pytest.mark.examples)
        
        if "api" in str(item.fspath) or "endpoint" in item.name.lower():
            item.add_marker(pytest.mark.api)
        
        if "e2e" in str(item.fspath) or "workflow" in item.name.lower():
            item.add_marker(pytest.mark.e2e)
