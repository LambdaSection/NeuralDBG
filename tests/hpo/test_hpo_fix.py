import os
import sys
import pytest

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.code_generation.code_generator import generate_optimized_dsl

class TestHPOFix:

    def test_generate_optimized_dsl_with_dict(self):
        """Test HPO parameter replacement when the parameter is a dictionary."""
        config = """
        network DictTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64, 128, 256)))
                Dropout(HPO(range(0.3, 0.7, step=0.1)))
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """

        # Dictionary value in best_params
        best_params = {
            'dense_units': {'value': 128},
            'dropout_rate': 0.5,
            'learning_rate': 0.001
        }

        # This should not raise an error
        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "DictTest" in optimized

        # Verify HPO expressions were replaced
        assert "HPO(choice(64, 128, 256))" not in optimized
        assert "HPO(range(0.3, 0.7, step=0.1))" not in optimized
        assert "HPO(log_range(1e-4, 1e-2))" not in optimized

        # Verify the dictionary value was properly handled
        assert "{'value': 128}" in optimized or '{"value": 128}' in optimized
        assert "0.5" in optimized
        assert "learning_rate=0.001" in optimized
