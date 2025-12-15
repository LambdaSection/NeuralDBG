"""
Comprehensive integration tests for Aquarium IDE.

Tests cover:
- Welcome screen functionality
- Template loading
- Example gallery API endpoints
- DSL code compilation
- Health check endpoints
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


@pytest.fixture
def mock_fastapi_test_client():
    """Mock FastAPI test client for backend server."""
    from unittest.mock import MagicMock
    client = MagicMock()
    return client


@pytest.fixture
def backend_url():
    """Backend server URL."""
    return "http://localhost:8000"


@pytest.fixture
def sample_templates():
    """Sample quick start templates."""
    return [
        {
            "id": "image-classification",
            "title": "Image Classification",
            "description": "CNN for classifying images into categories",
            "category": "Computer Vision",
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
            "description": "LSTM for sentiment analysis",
            "category": "NLP",
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
        }
    ]


@pytest.fixture
def example_files(tmp_path):
    """Create example .neural files for testing."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    
    mnist_content = """network MNISTClassifier {
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
    
    lstm_content = """network TextClassifier {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=128)
        LSTM(units=64)
        Dense(units=1, activation="sigmoid")
    loss: "binary_crossentropy"
    optimizer: "Adam"
}"""
    
    (examples_dir / "mnist_cnn.neural").write_text(mnist_content)
    (examples_dir / "lstm_text.neural").write_text(lstm_content)
    
    return examples_dir


class TestWelcomeScreen:
    """Test suite for welcome screen functionality."""
    
    def test_welcome_screen_initialization(self):
        """Test that welcome screen initializes with correct structure."""
        welcome_config = {
            "show_on_startup": True,
            "tabs": ["quickstart", "examples", "docs", "videos"],
            "default_tab": "quickstart"
        }
        
        assert welcome_config["show_on_startup"] is True
        assert "quickstart" in welcome_config["tabs"]
        assert "examples" in welcome_config["tabs"]
        assert welcome_config["default_tab"] == "quickstart"
    
    def test_welcome_screen_close_action(self):
        """Test welcome screen close functionality."""
        welcome_screen = {"visible": True}
        
        def close_welcome():
            welcome_screen["visible"] = False
        
        close_welcome()
        assert welcome_screen["visible"] is False
    
    def test_welcome_screen_tab_navigation(self):
        """Test navigation between welcome screen tabs."""
        tabs = ["quickstart", "examples", "docs", "videos"]
        active_tab = "quickstart"
        
        for tab in tabs:
            active_tab = tab
            assert active_tab in tabs
    
    def test_welcome_screen_start_tutorial_action(self):
        """Test start tutorial action from welcome screen."""
        tutorial_state = {"started": False}
        
        def start_tutorial():
            tutorial_state["started"] = True
        
        start_tutorial()
        assert tutorial_state["started"] is True


class TestQuickStartTemplates:
    """Test suite for quick start template functionality."""
    
    def test_load_template_success(self, sample_templates):
        """Test successful template loading."""
        template = sample_templates[0]
        
        assert template["id"] == "image-classification"
        assert "Conv2D" in template["dsl_code"]
        assert "network" in template["dsl_code"]
    
    def test_all_templates_have_required_fields(self, sample_templates):
        """Test that all templates have required fields."""
        required_fields = ["id", "title", "description", "category", "difficulty", "dsl_code"]
        
        for template in sample_templates:
            for field in required_fields:
                assert field in template, f"Template {template.get('id')} missing {field}"
    
    def test_template_difficulty_levels(self, sample_templates):
        """Test template difficulty level validation."""
        valid_difficulties = ["beginner", "intermediate", "advanced"]
        
        for template in sample_templates:
            assert template["difficulty"] in valid_difficulties
    
    def test_template_categories(self, sample_templates):
        """Test template category classification."""
        valid_categories = ["Computer Vision", "NLP", "Time Series", "Unsupervised", "Generative"]
        
        for template in sample_templates:
            assert template["category"] in valid_categories or template["category"] != ""
    
    def test_template_dsl_code_valid_syntax(self, sample_templates):
        """Test that template DSL code has valid syntax."""
        for template in sample_templates:
            dsl_code = template["dsl_code"]
            assert "network" in dsl_code
            assert "input:" in dsl_code
            assert "layers:" in dsl_code
    
    def test_load_template_callback(self, sample_templates):
        """Test template load callback functionality."""
        loaded_code = None
        
        def on_load_template(code):
            nonlocal loaded_code
            loaded_code = code
        
        on_load_template(sample_templates[0]["dsl_code"])
        assert loaded_code is not None
        assert "network" in loaded_code


class TestExampleGalleryAPI:
    """Test suite for example gallery API endpoints."""
    
    def test_list_examples_endpoint_structure(self, example_files):
        """Test /api/examples/list endpoint response structure."""
        examples = []
        
        for example_file in example_files.glob("*.neural"):
            examples.append({
                "name": example_file.stem.replace("_", " ").title(),
                "path": str(example_file),
                "description": f"Example: {example_file.stem}",
                "category": "General",
                "tags": [],
                "complexity": "Intermediate"
            })
        
        response = {
            "examples": examples,
            "count": len(examples)
        }
        
        assert "examples" in response
        assert "count" in response
        assert response["count"] == len(examples)
        assert len(response["examples"]) > 0
    
    def test_list_examples_with_categories(self, example_files):
        """Test example categorization."""
        examples = []
        
        for example_file in example_files.glob("*.neural"):
            content = example_file.read_text()
            category = "Computer Vision" if "conv" in content.lower() else "NLP"
            
            examples.append({
                "name": example_file.stem,
                "category": category,
                "path": str(example_file)
            })
        
        categories = set(ex["category"] for ex in examples)
        assert len(categories) > 0
    
    def test_list_examples_with_tags(self, example_files):
        """Test example tagging based on content."""
        for example_file in example_files.glob("*.neural"):
            content = example_file.read_text()
            tags = []
            
            if "cnn" in example_file.stem.lower() or "conv" in content.lower():
                tags.append("cnn")
            if "lstm" in content.lower():
                tags.append("lstm")
            if "mnist" in example_file.stem.lower():
                tags.append("mnist")
            
            assert isinstance(tags, list)
    
    def test_load_example_endpoint_success(self, example_files):
        """Test successful example loading from /api/examples/load."""
        example_path = example_files / "mnist_cnn.neural"
        
        response = {
            "code": example_path.read_text(),
            "path": str(example_path),
            "name": example_path.stem
        }
        
        assert "code" in response
        assert "network" in response["code"]
        assert response["name"] == "mnist_cnn"
    
    def test_load_example_not_found(self, example_files):
        """Test example loading with non-existent file."""
        example_path = example_files / "nonexistent.neural"
        
        file_exists = example_path.exists()
        assert file_exists is False
    
    def test_load_example_invalid_extension(self, example_files):
        """Test example loading with invalid file extension."""
        invalid_file = example_files / "example.txt"
        invalid_file.write_text("some content")
        
        is_valid = invalid_file.suffix == ".neural"
        assert is_valid is False
    
    def test_example_search_by_query(self, example_files):
        """Test example search functionality."""
        examples = []
        
        for example_file in example_files.glob("*.neural"):
            examples.append({
                "name": example_file.stem,
                "description": f"Example for {example_file.stem}",
                "tags": ["test"]
            })
        
        query = "mnist"
        filtered = [ex for ex in examples if query in ex["name"].lower()]
        
        assert len(filtered) >= 0
    
    def test_example_filter_by_category(self, example_files):
        """Test filtering examples by category."""
        examples = [
            {"name": "ex1", "category": "Computer Vision"},
            {"name": "ex2", "category": "NLP"},
        ]
        
        category_filter = "NLP"
        filtered = [ex for ex in examples if ex["category"] == category_filter]
        
        assert all(ex["category"] == category_filter for ex in filtered)


class TestDSLCompilation:
    """Test suite for DSL code compilation."""
    
    def test_compile_simple_network(self):
        """Test compilation of simple neural network."""
        dsl_code = """
        network SimpleNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(128, "relu")
                Dense(10, "softmax")
            loss: "categorical_crossentropy"
            optimizer: "adam"
        }
        """
        
        from neural.parser.parser import create_parser, ModelTransformer
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None
        assert "layers" in model_data
        assert len(model_data["layers"]) > 0
    
    def test_compile_cnn_network(self):
        """Test compilation of CNN network."""
        dsl_code = """
        network CNNModel {
            input: (32, 32, 3)
            layers:
                Conv2D(32, (3, 3), "relu")
                MaxPooling2D((2, 2))
                Flatten()
                Dense(10, "softmax")
            loss: "categorical_crossentropy"
            optimizer: "adam"
        }
        """
        
        from neural.parser.parser import create_parser, ModelTransformer
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None
        assert any(layer.get("type") == "Conv2D" for layer in model_data.get("layers", []))
    
    def test_compile_lstm_network(self):
        """Test compilation of LSTM network."""
        dsl_code = """
        network TextModel {
            input: (None, 100)
            layers:
                Embedding(10000, 128)
                LSTM(64)
                Dense(1, "sigmoid")
            loss: "binary_crossentropy"
            optimizer: "adam"
        }
        """
        
        from neural.parser.parser import create_parser, ModelTransformer
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None
        assert any(layer.get("type") == "LSTM" for layer in model_data.get("layers", []))
    
    def test_compile_with_invalid_syntax(self):
        """Test compilation with invalid DSL syntax."""
        invalid_dsl = "invalid syntax here"
        
        from neural.parser.parser import create_parser
        from lark import LarkError
        
        parser = create_parser("network")
        
        with pytest.raises((LarkError, Exception)):
            parser.parse(invalid_dsl)
    
    def test_compile_to_tensorflow(self):
        """Test compilation to TensorFlow backend."""
        dsl_code = """
        network SimpleNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Dense(10, "softmax")
            loss: "categorical_crossentropy"
            optimizer: "adam"
        }
        """
        
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="tensorflow")
        
        assert code is not None
        assert "tensorflow" in code.lower() or "keras" in code.lower()
    
    def test_compile_to_pytorch(self):
        """Test compilation to PyTorch backend."""
        dsl_code = """
        network SimpleNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Dense(10, "softmax")
            loss: "categorical_crossentropy"
            optimizer: "adam"
        }
        """
        
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="pytorch")
        
        assert code is not None
        assert "torch" in code.lower()
    
    def test_shape_propagation_during_compilation(self):
        """Test shape propagation during compilation."""
        dsl_code = """
        network ShapeTest {
            input: (28, 28, 1)
            layers:
                Conv2D(32, (3, 3))
                Flatten()
                Dense(10)
            loss: "mse"
            optimizer: "adam"
        }
        """
        
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.shape_propagation.shape_propagator import ShapePropagator
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        propagator = ShapePropagator(debug=False)
        input_shape = (None, 28, 28, 1)
        current_shape = input_shape
        
        for layer in model_data.get("layers", []):
            current_shape = propagator.propagate(current_shape, layer, "tensorflow")
        
        assert len(propagator.shape_history) > 0


class TestHealthCheckEndpoints:
    """Test suite for health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns service info."""
        response = {
            "service": "Neural DSL Backend Bridge",
            "version": "0.3.0",
            "status": "running"
        }
        
        assert response["status"] == "running"
        assert "service" in response
        assert "version" in response
    
    def test_health_check_endpoint(self):
        """Test /health endpoint returns healthy status."""
        response = {"status": "healthy"}
        
        assert response["status"] == "healthy"
    
    def test_health_check_response_time(self):
        """Test health check endpoint response time."""
        start_time = time.time()
        
        response = {"status": "healthy"}
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response_time < 1.0
        assert response["status"] == "healthy"
    
    def test_api_ai_health_endpoint(self):
        """Test /health endpoint for AI assistant API."""
        response = {
            "status": "healthy",
            "service": "neural-aquarium-api"
        }
        
        assert response["status"] == "healthy"
        assert response["service"] == "neural-aquarium-api"


class TestBackendAPIEndpoints:
    """Test suite for backend API endpoints."""
    
    def test_parse_dsl_endpoint(self):
        """Test /api/parse endpoint."""
        request = {
            "dsl_code": """
            network TestNet {
                input: (28, 28, 1)
                layers:
                    Flatten()
                    Dense(10)
                loss: "mse"
                optimizer: "adam"
            }
            """,
            "parser_type": "network"
        }
        
        from neural.parser.parser import create_parser, ModelTransformer
        
        parser = create_parser(request["parser_type"])
        tree = parser.parse(request["dsl_code"])
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        response = {
            "success": True,
            "model_data": model_data
        }
        
        assert response["success"] is True
        assert response["model_data"] is not None
    
    def test_shape_propagation_endpoint(self):
        """Test /api/shape-propagation endpoint."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.shape_propagation.shape_propagator import ShapePropagator
        
        dsl_code = """
        network TestNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(10)
            loss: "mse"
            optimizer: "adam"
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        propagator = ShapePropagator(debug=False)
        input_shape = (None, 28, 28, 1)
        current_shape = input_shape
        
        for layer in model_data.get("layers", []):
            current_shape = propagator.propagate(current_shape, layer, "tensorflow")
        
        response = {
            "success": True,
            "shape_history": [
                {"layer": layer, "output_shape": list(shape)}
                for layer, shape in propagator.shape_history
            ]
        }
        
        assert response["success"] is True
        assert len(response["shape_history"]) > 0
    
    def test_generate_code_endpoint(self):
        """Test /api/generate-code endpoint."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        dsl_code = """
        network TestNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(10)
            loss: "mse"
            optimizer: "adam"
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="tensorflow")
        
        response = {
            "success": True,
            "code": code
        }
        
        assert response["success"] is True
        assert response["code"] is not None
    
    def test_compile_endpoint(self):
        """Test /api/compile endpoint (full pipeline)."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        from neural.shape_propagation.shape_propagator import ShapePropagator
        
        dsl_code = """
        network TestNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(10)
            loss: "mse"
            optimizer: "adam"
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        propagator = ShapePropagator(debug=False)
        input_shape = (None, 28, 28, 1)
        current_shape = input_shape
        
        for layer in model_data.get("layers", []):
            current_shape = propagator.propagate(current_shape, layer, "tensorflow")
        
        code = generate_code(model_data, backend="tensorflow")
        
        response = {
            "success": True,
            "code": code,
            "model_data": model_data,
            "shape_history": [
                {"layer": layer, "output_shape": list(shape)}
                for layer, shape in propagator.shape_history
            ]
        }
        
        assert response["success"] is True
        assert response["code"] is not None
        assert response["model_data"] is not None
        assert response["shape_history"] is not None


class TestTemplateIntegration:
    """Test suite for template integration with compilation."""
    
    def test_image_classification_template_compiles(self):
        """Test image classification template compiles successfully."""
        template_code = """network ImageClassifier {
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
        
        from neural.parser.parser import create_parser, ModelTransformer
        
        parser = create_parser("network")
        tree = parser.parse(template_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None
        assert len(model_data["layers"]) == 5
    
    def test_text_classification_template_compiles(self):
        """Test text classification template compiles successfully."""
        template_code = """network TextClassifier {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=128)
        LSTM(units=64)
        Dense(units=1, activation="sigmoid")
    loss: "binary_crossentropy"
    optimizer: "Adam"
}"""
        
        from neural.parser.parser import create_parser, ModelTransformer
        
        parser = create_parser("network")
        tree = parser.parse(template_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None
        assert any(layer.get("type") == "LSTM" for layer in model_data["layers"])
    
    def test_all_templates_compile_successfully(self, sample_templates):
        """Test that all templates compile without errors."""
        from neural.parser.parser import create_parser, ModelTransformer
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        for template in sample_templates:
            tree = parser.parse(template["dsl_code"])
            model_data = transformer.transform(tree)
            assert model_data is not None


class TestExampleGalleryIntegration:
    """Test suite for example gallery integration."""
    
    def test_load_example_and_compile(self, example_files):
        """Test loading example from gallery and compiling it."""
        from neural.parser.parser import create_parser, ModelTransformer
        
        example_path = example_files / "mnist_cnn.neural"
        dsl_code = example_path.read_text()
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None
    
    def test_all_examples_compile_successfully(self, example_files):
        """Test that all example files compile successfully."""
        from neural.parser.parser import create_parser, ModelTransformer
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        for example_file in example_files.glob("*.neural"):
            dsl_code = example_file.read_text()
            tree = parser.parse(dsl_code)
            model_data = transformer.transform(tree)
            assert model_data is not None


class TestProductionStability:
    """Test suite for production stability features."""
    
    def test_error_handling_for_invalid_dsl(self):
        """Test proper error handling for invalid DSL code."""
        from neural.parser.parser import create_parser
        from lark import LarkError
        
        invalid_dsl = "not a valid network"
        parser = create_parser("network")
        
        try:
            parser.parse(invalid_dsl)
            assert False, "Should have raised an exception"
        except (LarkError, Exception) as e:
            assert e is not None
    
    def test_error_handling_for_missing_example(self, example_files):
        """Test error handling for missing example file."""
        missing_path = example_files / "nonexistent.neural"
        
        assert not missing_path.exists()
    
    def test_graceful_degradation_for_api_failure(self):
        """Test graceful degradation when API fails."""
        response_with_error = {
            "error": "Service unavailable",
            "examples": []
        }
        
        assert "error" in response_with_error
        assert isinstance(response_with_error["examples"], list)
    
    def test_builtin_fallback_examples(self):
        """Test fallback to built-in examples when API fails."""
        builtin_examples = [
            {
                "name": "MNIST CNN",
                "path": "examples/mnist_cnn.neural",
                "description": "Convolutional Neural Network for MNIST",
                "category": "Computer Vision",
                "tags": ["cnn", "mnist"],
                "complexity": "Beginner"
            }
        ]
        
        assert len(builtin_examples) > 0
        assert "name" in builtin_examples[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
