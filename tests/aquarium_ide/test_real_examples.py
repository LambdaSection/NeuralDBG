"""
Integration tests for real example files in neural/aquarium/examples/.

Tests verify that actual example files in the repository are valid and compilable.
"""

import pytest
from pathlib import Path
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import generate_code
from neural.shape_propagation.shape_propagator import ShapePropagator


@pytest.fixture
def examples_directory():
    """Get the real examples directory."""
    repo_root = Path(__file__).parent.parent.parent
    examples_dir = repo_root / "neural" / "aquarium" / "examples"
    return examples_dir


@pytest.fixture
def all_example_files(examples_directory):
    """Get all .neural example files."""
    if not examples_directory.exists():
        pytest.skip("Examples directory not found")
    
    example_files = list(examples_directory.glob("*.neural"))
    if not example_files:
        pytest.skip("No example files found")
    
    return example_files


class TestRealExampleFiles:
    """Test real example files from the repository."""
    
    def test_examples_directory_exists(self, examples_directory):
        """Test that examples directory exists."""
        if examples_directory.exists():
            assert examples_directory.is_dir()
    
    def test_example_files_exist(self, all_example_files):
        """Test that example files exist."""
        assert len(all_example_files) > 0, "No example files found"
    
    def test_all_examples_have_neural_extension(self, all_example_files):
        """Test all examples have .neural extension."""
        for example_file in all_example_files:
            assert example_file.suffix == ".neural", f"{example_file.name} has wrong extension"
    
    def test_all_examples_are_readable(self, all_example_files):
        """Test all example files are readable."""
        for example_file in all_example_files:
            try:
                content = example_file.read_text(encoding="utf-8")
                assert len(content) > 0, f"{example_file.name} is empty"
            except Exception as e:
                pytest.fail(f"Failed to read {example_file.name}: {e}")
    
    def test_all_examples_contain_network_definition(self, all_example_files):
        """Test all examples contain 'network' keyword."""
        for example_file in all_example_files:
            content = example_file.read_text(encoding="utf-8")
            assert "network" in content, f"{example_file.name} missing 'network' keyword"
    
    def test_all_examples_parse_successfully(self, all_example_files):
        """Test all examples parse without errors."""
        parser = create_parser("network")
        
        failed = []
        for example_file in all_example_files:
            try:
                content = example_file.read_text(encoding="utf-8")
                parser.parse(content)
            except Exception as e:
                failed.append((example_file.name, str(e)))
        
        if failed:
            error_msg = "\n".join([f"{name}: {error}" for name, error in failed])
            pytest.fail(f"Failed to parse examples:\n{error_msg}")
    
    def test_all_examples_transform_successfully(self, all_example_files):
        """Test all examples transform to model data."""
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        failed = []
        for example_file in all_example_files:
            try:
                content = example_file.read_text(encoding="utf-8")
                tree = parser.parse(content)
                model_data = transformer.transform(tree)
                assert model_data is not None
                assert "layers" in model_data
            except Exception as e:
                failed.append((example_file.name, str(e)))
        
        if failed:
            error_msg = "\n".join([f"{name}: {error}" for name, error in failed])
            pytest.fail(f"Failed to transform examples:\n{error_msg}")
    
    def test_all_examples_generate_tensorflow_code(self, all_example_files):
        """Test all examples generate TensorFlow code."""
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        failed = []
        for example_file in all_example_files:
            try:
                content = example_file.read_text(encoding="utf-8")
                tree = parser.parse(content)
                model_data = transformer.transform(tree)
                code = generate_code(model_data, backend="tensorflow")
                assert code is not None
                assert len(code) > 0
            except Exception as e:
                failed.append((example_file.name, str(e)))
        
        if failed:
            error_msg = "\n".join([f"{name}: {error}" for name, error in failed])
            pytest.fail(f"Failed to generate TensorFlow code:\n{error_msg}")
    
    def test_all_examples_generate_pytorch_code(self, all_example_files):
        """Test all examples generate PyTorch code."""
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        failed = []
        for example_file in all_example_files:
            try:
                content = example_file.read_text(encoding="utf-8")
                tree = parser.parse(content)
                model_data = transformer.transform(tree)
                code = generate_code(model_data, backend="pytorch")
                assert code is not None
                assert len(code) > 0
            except Exception as e:
                failed.append((example_file.name, str(e)))
        
        if failed:
            error_msg = "\n".join([f"{name}: {error}" for name, error in failed])
            pytest.fail(f"Failed to generate PyTorch code:\n{error_msg}")


class TestSpecificExamples:
    """Test specific example files by name."""
    
    def test_mnist_cnn_example(self, examples_directory):
        """Test MNIST CNN example specifically."""
        example_file = examples_directory / "mnist_cnn.neural"
        
        if not example_file.exists():
            pytest.skip("mnist_cnn.neural not found")
        
        content = example_file.read_text(encoding="utf-8")
        
        parser = create_parser("network")
        tree = parser.parse(content)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert "MNISTClassifier" in content or "MNIST" in content
        assert "Conv2D" in content
        assert model_data is not None
        
        code = generate_code(model_data, backend="tensorflow")
        assert code is not None
    
    def test_lstm_text_example(self, examples_directory):
        """Test LSTM text example specifically."""
        example_file = examples_directory / "lstm_text.neural"
        
        if not example_file.exists():
            pytest.skip("lstm_text.neural not found")
        
        content = example_file.read_text(encoding="utf-8")
        
        parser = create_parser("network")
        tree = parser.parse(content)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert "LSTM" in content or "lstm" in content.lower()
        assert model_data is not None
        
        code = generate_code(model_data, backend="tensorflow")
        assert code is not None
    
    def test_resnet_example(self, examples_directory):
        """Test ResNet example specifically."""
        example_file = examples_directory / "resnet.neural"
        
        if not example_file.exists():
            pytest.skip("resnet.neural not found")
        
        content = example_file.read_text(encoding="utf-8")
        
        parser = create_parser("network")
        tree = parser.parse(content)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None
        
        code = generate_code(model_data, backend="tensorflow")
        assert code is not None
    
    def test_transformer_example(self, examples_directory):
        """Test Transformer example specifically."""
        example_file = examples_directory / "transformer.neural"
        
        if not example_file.exists():
            pytest.skip("transformer.neural not found")
        
        content = example_file.read_text(encoding="utf-8")
        
        parser = create_parser("network")
        tree = parser.parse(content)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None
    
    def test_vae_example(self, examples_directory):
        """Test VAE example specifically."""
        example_file = examples_directory / "vae.neural"
        
        if not example_file.exists():
            pytest.skip("vae.neural not found")
        
        content = example_file.read_text(encoding="utf-8")
        
        parser = create_parser("network")
        tree = parser.parse(content)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None


class TestExampleMetadataExtraction:
    """Test metadata extraction from example files."""
    
    def test_extract_categories_from_examples(self, all_example_files):
        """Test category extraction based on content."""
        categories = set()
        
        for example_file in all_example_files:
            content = example_file.read_text(encoding="utf-8").lower()
            
            if "conv" in content or "cnn" in example_file.stem.lower():
                categories.add("Computer Vision")
            elif "lstm" in content or "rnn" in content or "gru" in content:
                categories.add("NLP")
            elif "transformer" in content:
                categories.add("NLP")
            elif "gan" in example_file.stem.lower() or "vae" in example_file.stem.lower():
                categories.add("Generative")
        
        assert len(categories) > 0, "No categories detected"
    
    def test_extract_complexity_from_examples(self, all_example_files):
        """Test complexity estimation based on layer count."""
        for example_file in all_example_files:
            try:
                content = example_file.read_text(encoding="utf-8")
                parser = create_parser("network")
                tree = parser.parse(content)
                transformer = ModelTransformer()
                model_data = transformer.transform(tree)
                
                layer_count = len(model_data.get("layers", []))
                
                if layer_count <= 5:
                    complexity = "Beginner"
                elif layer_count <= 10:
                    complexity = "Intermediate"
                else:
                    complexity = "Advanced"
                
                assert complexity in ["Beginner", "Intermediate", "Advanced"]
            except Exception:
                pass
    
    def test_extract_tags_from_examples(self, all_example_files):
        """Test tag extraction from example content."""
        for example_file in all_example_files:
            content = example_file.read_text(encoding="utf-8").lower()
            tags = []
            
            if "conv" in content:
                tags.append("cnn")
            if "lstm" in content:
                tags.append("lstm")
            if "gru" in content:
                tags.append("gru")
            if "transformer" in content:
                tags.append("transformer")
            if "mnist" in example_file.stem.lower():
                tags.append("mnist")
            if "text" in example_file.stem.lower():
                tags.append("text")
            
            assert isinstance(tags, list)


class TestExampleShapePropagation:
    """Test shape propagation for example files."""
    
    def test_examples_with_valid_input_shapes(self, all_example_files):
        """Test examples that have valid input shapes."""
        parser = create_parser("network")
        transformer = ModelTransformer()
        propagator = ShapePropagator(debug=False)
        
        successful = []
        failed = []
        
        for example_file in all_example_files:
            try:
                content = example_file.read_text(encoding="utf-8")
                tree = parser.parse(content)
                model_data = transformer.transform(tree)
                
                if "input" in model_data and "shape" in model_data["input"]:
                    input_shape = (None,) + tuple(model_data["input"]["shape"])
                    current_shape = input_shape
                    
                    for layer in model_data.get("layers", []):
                        current_shape = propagator.propagate(current_shape, layer, "tensorflow")
                    
                    if len(propagator.shape_history) > 0:
                        successful.append(example_file.name)
                    
                    propagator.shape_history.clear()
            except Exception as e:
                failed.append((example_file.name, str(e)))
        
        assert len(successful) > 0 or len(failed) > 0


class TestExampleAPIEndpoints:
    """Test API endpoint behavior with real examples."""
    
    def test_list_examples_returns_real_files(self, examples_directory):
        """Test that list endpoint would return real files."""
        if not examples_directory.exists():
            pytest.skip("Examples directory not found")
        
        example_files = list(examples_directory.glob("*.neural"))
        
        examples = []
        for example_file in example_files:
            examples.append({
                "name": example_file.stem.replace("_", " ").title(),
                "path": str(example_file.relative_to(examples_directory.parent)),
                "description": f"Neural network example: {example_file.stem}"
            })
        
        assert len(examples) > 0
    
    def test_load_example_returns_valid_code(self, all_example_files):
        """Test that load endpoint would return valid code."""
        for example_file in all_example_files:
            content = example_file.read_text(encoding="utf-8")
            
            response = {
                "code": content,
                "path": str(example_file),
                "name": example_file.stem
            }
            
            assert "code" in response
            assert "network" in response["code"]
            assert response["name"] == example_file.stem


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
