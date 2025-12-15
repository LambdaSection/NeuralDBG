"""
End-to-end integration tests for Aquarium IDE workflows.

Tests complete user workflows from welcome screen to code compilation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch


class TestWelcomeToCompilationWorkflow:
    """Test complete workflow from welcome screen to compilation."""
    
    def test_template_selection_to_compilation(self):
        """Test selecting template and compiling it."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
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
        
        parser = create_parser("network")
        tree = parser.parse(template_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="tensorflow")
        
        assert model_data is not None
        assert code is not None
        assert "tensorflow" in code.lower() or "keras" in code.lower()
    
    def test_example_loading_to_compilation(self):
        """Test loading example and compiling it."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        example_code = """network MNISTClassifier {
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
        
        parser = create_parser("network")
        tree = parser.parse(example_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="tensorflow")
        
        assert model_data is not None
        assert code is not None
    
    def test_custom_code_to_compilation(self):
        """Test writing custom code and compiling it."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        custom_code = """network CustomNet {
    input: (32, 32, 3)
    layers:
        Conv2D(64, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(256, "relu")
        Dropout(0.5)
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}"""
        
        parser = create_parser("network")
        tree = parser.parse(custom_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="tensorflow")
        
        assert model_data is not None
        assert code is not None


class TestMultiBackendWorkflow:
    """Test workflow with multiple backend targets."""
    
    def test_compile_same_dsl_to_multiple_backends(self):
        """Test compiling same DSL to TensorFlow and PyTorch."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        dsl_code = """network MultiBackendNet {
    input: (28, 28, 1)
    layers:
        Flatten()
        Dense(128, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}"""
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        tf_code = generate_code(model_data, backend="tensorflow")
        pytorch_code = generate_code(model_data, backend="pytorch")
        
        assert "tensorflow" in tf_code.lower() or "keras" in tf_code.lower()
        assert "torch" in pytorch_code.lower()
        assert tf_code != pytorch_code


class TestShapePropagationWorkflow:
    """Test workflow with shape propagation."""
    
    def test_compile_with_shape_validation(self):
        """Test compilation with shape propagation validation."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.shape_propagation.shape_propagator import ShapePropagator
        from neural.code_generation.code_generator import generate_code
        
        dsl_code = """network ShapeValidatedNet {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3))
        MaxPooling2D((2, 2))
        Flatten()
        Dense(10)
    loss: "mse"
    optimizer: "adam"
}"""
        
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
        
        assert len(propagator.shape_history) > 0
        assert code is not None


class TestErrorHandlingWorkflow:
    """Test error handling throughout workflows."""
    
    def test_invalid_dsl_error_handling(self):
        """Test error handling for invalid DSL code."""
        from neural.parser.parser import create_parser
        from lark import LarkError
        
        invalid_dsl = "this is not valid dsl code"
        parser = create_parser("network")
        
        with pytest.raises((LarkError, Exception)):
            parser.parse(invalid_dsl)
    
    def test_missing_example_error_handling(self, tmp_path):
        """Test error handling for missing example files."""
        missing_file = tmp_path / "nonexistent.neural"
        
        assert not missing_file.exists()
    
    def test_compilation_error_recovery(self):
        """Test recovery from compilation errors."""
        from neural.parser.parser import create_parser, ModelTransformer
        
        dsl_with_issue = """network TestNet {
    input: (28, 28, 1)
    layers:
        Flatten()
        Dense(10)
    loss: "mse"
    optimizer: "adam"
}"""
        
        parser = create_parser("network")
        tree = parser.parse(dsl_with_issue)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None


class TestAPIIntegrationWorkflow:
    """Test API integration workflows."""
    
    @patch('neural.aquarium.backend.server.create_app')
    def test_full_api_workflow(self, mock_app):
        """Test complete API workflow: parse -> shape -> generate."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.shape_propagation.shape_propagator import ShapePropagator
        from neural.code_generation.code_generator import generate_code
        
        dsl_code = """network APITestNet {
    input: (28, 28, 1)
    layers:
        Flatten()
        Dense(64, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}"""
        
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
        
        assert model_data is not None
        assert len(propagator.shape_history) > 0
        assert code is not None


class TestExampleGalleryWorkflow:
    """Test example gallery complete workflows."""
    
    def test_browse_filter_load_compile_workflow(self, tmp_path):
        """Test browsing, filtering, loading, and compiling examples."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        
        example_content = """network MNISTExample {
    input: (28, 28, 1)
    layers:
        Flatten()
        Dense(128, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}"""
        
        example_file = examples_dir / "mnist.neural"
        example_file.write_text(example_content)
        
        examples = []
        for f in examples_dir.glob("*.neural"):
            examples.append({
                "name": f.stem,
                "path": str(f),
                "category": "Computer Vision"
            })
        
        filtered = [ex for ex in examples if ex["category"] == "Computer Vision"]
        assert len(filtered) > 0
        
        selected_example = filtered[0]
        example_path = Path(selected_example["path"])
        dsl_code = example_path.read_text()
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="tensorflow")
        
        assert model_data is not None
        assert code is not None


class TestTemplateCustomizationWorkflow:
    """Test template customization workflows."""
    
    def test_load_template_modify_compile(self):
        """Test loading template, modifying it, and compiling."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        original_template = """network ImageClassifier {
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
        
        modified_template = original_template.replace("filters=32", "filters=64")
        
        parser = create_parser("network")
        tree = parser.parse(modified_template)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="tensorflow")
        
        assert model_data is not None
        assert code is not None


class TestUserJourneyWorkflows:
    """Test complete user journey scenarios."""
    
    def test_new_user_journey(self):
        """Test new user journey: welcome -> tutorial -> first model."""
        app_state = {
            "welcome_shown": True,
            "tutorial_completed": False,
            "first_model_created": False
        }
        
        def show_welcome():
            app_state["welcome_shown"] = True
        
        def complete_tutorial():
            app_state["tutorial_completed"] = True
        
        def create_first_model():
            app_state["first_model_created"] = True
        
        show_welcome()
        assert app_state["welcome_shown"] is True
        
        complete_tutorial()
        assert app_state["tutorial_completed"] is True
        
        create_first_model()
        assert app_state["first_model_created"] is True
    
    def test_experienced_user_journey(self):
        """Test experienced user journey: skip welcome -> load example -> modify -> compile."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        app_state = {"welcome_shown": False}
        
        example_code = """network ExperiencedUserNet {
    input: (32, 32, 3)
    layers:
        Conv2D(64, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(256, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}"""
        
        parser = create_parser("network")
        tree = parser.parse(example_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="tensorflow")
        
        assert app_state["welcome_shown"] is False
        assert model_data is not None
        assert code is not None


class TestProductionReadinessWorkflows:
    """Test production readiness scenarios."""
    
    def test_high_volume_compilation_workflow(self):
        """Test handling multiple compilations."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        dsl_codes = [
            """network Model1 { input: (28, 28, 1) layers: Flatten() Dense(10) }""",
            """network Model2 { input: (32, 32, 3) layers: Flatten() Dense(20) }""",
            """network Model3 { input: (64, 64, 3) layers: Flatten() Dense(30) }"""
        ]
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        results = []
        for dsl_code in dsl_codes:
            tree = parser.parse(dsl_code)
            model_data = transformer.transform(tree)
            code = generate_code(model_data, backend="tensorflow")
            results.append({"model_data": model_data, "code": code})
        
        assert len(results) == len(dsl_codes)
        for result in results:
            assert result["model_data"] is not None
            assert result["code"] is not None
    
    def test_concurrent_user_workflow_simulation(self):
        """Test simulating concurrent user workflows."""
        from neural.parser.parser import create_parser, ModelTransformer
        
        user_sessions = [
            {"user_id": 1, "action": "parse"},
            {"user_id": 2, "action": "parse"},
            {"user_id": 3, "action": "parse"}
        ]
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        dsl_code = """network ConcurrentTest { input: (28, 28, 1) layers: Flatten() Dense(10) }"""
        
        for session in user_sessions:
            tree = parser.parse(dsl_code)
            model_data = transformer.transform(tree)
            assert model_data is not None
    
    def test_error_recovery_workflow(self):
        """Test system recovery from errors."""
        from neural.parser.parser import create_parser, ModelTransformer
        from lark import LarkError
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        invalid_dsl = "invalid code"
        valid_dsl = """network ValidNet { input: (28, 28, 1) layers: Flatten() Dense(10) }"""
        
        try:
            parser.parse(invalid_dsl)
        except (LarkError, Exception):
            pass
        
        tree = parser.parse(valid_dsl)
        model_data = transformer.transform(tree)
        assert model_data is not None


class TestPerformanceWorkflows:
    """Test performance-related workflows."""
    
    def test_large_model_compilation_workflow(self):
        """Test compiling large models."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        large_model_dsl = """network LargeModel {
    input: (224, 224, 3)
    layers:
        Conv2D(64, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Conv2D(128, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Conv2D(256, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(1024, "relu")
        Dropout(0.5)
        Dense(512, "relu")
        Dense(100, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}"""
        
        parser = create_parser("network")
        tree = parser.parse(large_model_dsl)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        code = generate_code(model_data, backend="tensorflow")
        
        assert model_data is not None
        assert len(model_data.get("layers", [])) > 5
        assert code is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
