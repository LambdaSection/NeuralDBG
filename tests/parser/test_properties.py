"""
Property-based tests for the Neural parser.

Tests invariants and properties that should always hold true.
Uses the correct Neural DSL syntax: network Name { input: ... layers: ... }
"""

from typing import Any, Dict  # Type hints

from hypothesis import given, settings  # Property-based testing
from hypothesis import strategies as st
import pytest  # Test framework for running tests

from neural.parser.parser import DSLValidationError, NeuralParser  # Parser classes


def verify_layer_structure(layer: Dict[str, Any]) -> bool:
    """
    Verify that a layer dictionary has the required structure.

    Args:
        layer: Dictionary representing a parsed layer.

    Returns:
        bool: True if layer has required 'type' key.
    """
    # Layers must have a 'type' field (e.g., 'Dense', 'Conv2D')
    required_keys = {'type'}
    return all(key in layer for key in required_keys)


def verify_model_structure(model: Dict[str, Any]) -> bool:
    """
    Verify that a parsed model has the required structure.

    Args:
        model: Dictionary representing a parsed network.

    Returns:
        bool: True if model has required 'input' and 'layers' keys.
    """
    # Model must be a dictionary
    if not isinstance(model, dict):
        return False

    # Required keys: 'input' for input shape, 'layers' for network layers
    required_keys = {'input', 'layers'}
    if not all(key in model for key in required_keys):
        return False

    # 'layers' must be a list
    if not isinstance(model['layers'], list):
        return False

    # Each layer must have valid structure
    return all(verify_layer_structure(layer) for layer in model['layers'])


class TestParserProperties:
    """
    Property-based tests for the Neural parser.

    Uses the correct Neural DSL syntax:
        network NetworkName {
            input: (dim1, dim2, ...)
            layers: Layer1(...) -> Layer2(...) -> ...
        }
    """

    def setup_method(self):
        """Initialize parser before each test."""
        self.parser = NeuralParser()  # Create new parser instance

    def test_simple_network_parsing(self):
        """Test that a simple network parses correctly."""
        # Valid Neural DSL syntax with network wrapper
        program = """
        network TestNet {
            input: (28, 28, 1)
            layers: Flatten() Dense(10)
        }
        """
        result = self.parser.parse(program)  # Parse the network
        # Verify required fields exist
        assert 'input' in result, "Parsed result must have 'input' field"
        assert 'layers' in result, "Parsed result must have 'layers' field"
        assert 'name' in result, "Parsed result must have 'name' field"
        assert result['name'] == 'TestNet', "Network name should be 'TestNet'"

    def test_input_shape_parsing(self):
        """Test that input shapes are parsed correctly."""
        # Network with multi-dimensional input
        program = """
        network ShapeTest {
            input: (None, 224, 224, 3)
            layers: Flatten() Dense(10)
        }
        """
        result = self.parser.parse(program)
        # Input should be a dict with 'type' and 'shape'
        assert 'input' in result
        assert result['input']['type'] == 'Input'
        # Shape should be a tuple
        assert result['input']['shape'] == (None, 224, 224, 3)

    def test_dense_layer_parsing(self):
        """Test that Dense layers parse correctly."""
        # Network with Dense layer including activation
        program = """
        network DenseTest {
            input: (100,)
            layers: Dense(64, "relu")
        }
        """
        result = self.parser.parse(program)
        assert verify_model_structure(result)
        # Check the Dense layer was parsed
        assert len(result['layers']) == 1
        assert result['layers'][0]['type'] == 'Dense'
        assert result['layers'][0]['params']['units'] == 64
        assert result['layers'][0]['params']['activation'] == 'relu'

    def test_multi_layer_network(self):
        """Test parsing a network with multiple layers."""
        # Network with layer chain using -> syntax
        program = """
        network MultiLayer {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(128, "relu")
                Dense(10, "softmax")
        }
        """
        result = self.parser.parse(program)
        assert verify_model_structure(result)
        # Should have 3 layers: Flatten, Dense, Dense
        assert len(result['layers']) == 3
        assert result['layers'][0]['type'] == 'Flatten'
        assert result['layers'][1]['type'] == 'Dense'
        assert result['layers'][2]['type'] == 'Dense'

    def test_invalid_syntax_raises_error(self):
        """Test that invalid syntax raises DSLValidationError."""
        # Invalid: no 'network' keyword
        invalid_program = """
        input: (28, 28)
        layers: Dense(10)
        """
        with pytest.raises(DSLValidationError):
            self.parser.parse(invalid_program)

    @settings(max_examples=10)  # Limit examples for speed
    @given(st.text(max_size=50))
    def test_parser_robustness(self, random_text: str):
        """
        Test that the parser handles arbitrary input without crashing.

        Should raise DSLValidationError for invalid input, not crash.
        """
        try:
            self.parser.parse(random_text)
        except DSLValidationError:
            # Expected for invalid input - parser correctly rejects garbage
            pass
        except Exception as e:
            # Other exceptions should mention parsing context
            assert "parse" in str(e).lower() or "error" in str(e).lower()