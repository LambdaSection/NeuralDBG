import os
import re
import sys


# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from neural.code_generation.code_generator import generate_optimized_dsl


class TestHPOCodeGeneration:

    def test_generate_optimized_dsl_basic(self):
        """Test basic HPO parameter replacement in DSL."""
        config = """
        network BasicTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64, 128, 256)))
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=0.001)
        }
        """

        best_params = {
            'dense_units': 128
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify HPO expression was replaced
        assert "HPO(choice(64, 128, 256))" not in optimized
        assert "Dense(128)" in optimized

    def test_generate_optimized_dsl_learning_rate(self):
        """Test learning rate HPO parameter replacement."""
        config = """
        network LRTest {
            input: (28, 28, 1)
            layers:
                Dense(128)
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """

        best_params = {
            'learning_rate': 0.001
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify HPO expression was replaced
        assert "HPO(log_range(1e-4, 1e-2))" not in optimized
        assert "learning_rate=0.001" in optimized

    def test_generate_optimized_dsl_multiple_params(self):
        """Test multiple HPO parameter replacements."""
        config = """
        network MultiTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64, 128, 256)))
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """

        best_params = {
            'dense_units': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify HPO expressions were replaced
        assert "HPO(choice(64, 128, 256))" not in optimized
        assert "HPO(range(0.2, 0.5, step=0.1))" not in optimized
        assert "HPO(log_range(1e-4, 1e-2))" not in optimized

        assert "Dense(128)" in optimized
        assert "Dropout(0.3)" in optimized
        assert "learning_rate=0.001" in optimized

    def test_generate_optimized_dsl_missing_params(self):
        """Test handling of missing parameters in best_params."""
        config = """
        network MissingTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64, 128, 256)))
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """

        # Missing dropout_rate
        best_params = {
            'dense_units': 128,
            'learning_rate': 0.001
        }

        # Should log a warning but not fail
        optimized = generate_optimized_dsl(config, best_params)

        # Verify available params were replaced
        assert "Dense(128)" in optimized
        assert "learning_rate=0.001" in optimized

        # Dropout HPO should still be present
        assert "HPO(range(0.2, 0.5, step=0.1))" in optimized

    def test_generate_optimized_dsl_nonexistent_param(self):
        """Test handling of nonexistent parameters."""
        config = """
        network InvalidTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64, 128)))
                Output(10)
            loss: "categorical_crossentropy"
        }
        """

        # Parameter name doesn't match what's in the config
        best_params = {
            'nonexistent_param': 128
        }

        # Should not fail
        optimized = generate_optimized_dsl(config, best_params)

        # Original HPO should still be present
        assert "HPO(choice(64, 128))" in optimized

    def test_generate_optimized_dsl_quoted_strings(self):
        """Test handling of quoted strings in HPO expressions."""
        config = """
        network QuotedTest {
            input: (28, 28, 1)
            layers:
                Dense(128, activation=HPO(choice("relu", "tanh", "sigmoid")))
                Output(10)
            loss: "categorical_crossentropy"
        }
        """

        best_params = {
            'dense_activation': "relu"
        }

        # This test is just to verify the code doesn't crash with quoted strings
        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "QuotedTest" in optimized

        # Verify the string value was properly handled (with quotes)
        assert 'activation="relu"' in optimized

    def test_generate_optimized_dsl_with_dict_values(self):
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

    def test_generate_optimized_dsl_with_complex_model(self):
        """Test HPO parameter replacement in a complex model with multiple layers and parameters."""
        config = """
        network ComplexTest {
            input: (224, 224, 3)
            layers:
                Conv2D(HPO(choice(32, 64)), kernel_size=(3, 3), activation=HPO(choice("relu", "elu")))
                MaxPooling2D(pool_size=(2, 2))
                Conv2D(HPO(choice(64, 128)), kernel_size=(3, 3), activation="relu")
                MaxPooling2D(pool_size=(2, 2))
                Flatten()
                Dense(HPO(range(128, 512, step=128)), activation="relu")
                Dropout(HPO(range(0.3, 0.7, step=0.1)))
                Dense(HPO(choice(64, 128)), activation="relu")
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Output(10, activation="softmax")
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
            train {
                epochs: HPO(choice(10, 20, 30))
                batch_size: HPO(choice(16, 32, 64))
            }
        }
        """

        best_params = {
            'conv2d_filters_1': 64,
            'conv2d_activation_1': 'relu',
            'conv2d_filters_2': 128,
            'dense_units_1': 256,
            'dropout_rate_1': 0.5,
            'dense_units_2': 128,
            'dropout_rate_2': 0.3,
            'learning_rate': 0.001,
            'epochs': 20,
            'batch_size': 32
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "ComplexTest" in optimized

        # Verify all HPO expressions were replaced
        assert "HPO(" not in optimized

        # Verify the values were properly handled
        assert re.search(r'Conv2D\(64,\s*kernel_size=\(3,\s*3\),\s*activation="relu"\)', optimized) or \
               'Conv2D(64, kernel_size=(3, 3), activation="relu")' in optimized
        assert re.search(r'Conv2D\(128,\s*kernel_size=\(3,\s*3\),\s*activation="relu"\)', optimized) or \
               'Conv2D(128, kernel_size=(3, 3), activation="relu")' in optimized
        assert "Dense(256" in optimized
        assert "Dropout(0.5" in optimized
        assert "Dense(128" in optimized
        assert "Dropout(0.3" in optimized
        assert "learning_rate=0.001" in optimized
        assert "epochs: 20" in optimized
        assert "batch_size: 32" in optimized

    def test_generate_optimized_dsl_with_nested_structures(self):
        """Test HPO parameter replacement in nested structures like tuples and lists."""
        config = """
        network NestedTest {
            input: (28, 28, 1)
            layers:
                Conv2D(32, kernel_size=(HPO(choice(3, 5)), HPO(choice(3, 5))))
                MaxPooling2D(pool_size=(HPO(choice(2, 3)), HPO(choice(2, 3))))
                Flatten()
                Dense(128)
                Dropout(0.5)
                Output(10)
            optimizer: Adam(learning_rate=0.001)
        }
        """

        best_params = {
            'conv2d_kernel_size_0': 3,
            'conv2d_kernel_size_1': 5,
            'maxpooling2d_pool_size_0': 2,
            'maxpooling2d_pool_size_1': 3
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "NestedTest" in optimized

        # Verify HPO expressions were replaced
        assert "HPO(choice(3, 5))" not in optimized
        assert "HPO(choice(2, 3))" not in optimized

        # Verify the nested values were properly handled
        assert "kernel_size=(3, 5)" in optimized
        assert "pool_size=(2, 3)" in optimized

    def test_generate_optimized_dsl_with_different_hpo_types(self):
        """Test HPO parameter replacement with different HPO types (choice, range, log_range)."""
        config = """
        network HPOTypesTest {
            input: (28, 28, 1)
            layers:
                Conv2D(HPO(choice(32, 64, 128)), kernel_size=(3, 3))
                MaxPooling2D(pool_size=(2, 2))
                Flatten()
                Dense(HPO(range(64, 256, step=64)))
                Dropout(HPO(range(0.2, 0.8, step=0.1)))
                Output(10)
            optimizer: Adam(learning_rate=HPO(log_range(1e-5, 1e-2)))
        }
        """

        best_params = {
            'conv2d_filters': 64,
            'dense_units': 128,
            'dropout_rate': 0.5,
            'learning_rate': 0.001
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "HPOTypesTest" in optimized

        # Verify HPO expressions were replaced
        assert "HPO(choice(32, 64, 128))" not in optimized
        assert "HPO(range(64, 256, step=64))" not in optimized
        assert "HPO(range(0.2, 0.8, step=0.1))" not in optimized
        assert "HPO(log_range(1e-5, 1e-2))" not in optimized

        # Verify the values were properly handled
        assert "Conv2D(64" in optimized
        assert "Dense(128" in optimized
        assert "Dropout(0.5" in optimized
        assert "learning_rate=0.001" in optimized

    def test_generate_optimized_dsl_with_multiple_optimizers(self):
        """Test HPO parameter replacement with different optimizer types."""
        config = """
        network OptimizerTest {
            input: (28, 28, 1)
            layers:
                Dense(128)
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: HPO(choice("Adam", "SGD", "RMSprop"))(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """

        best_params = {
            'optimizer_type': 'Adam',
            'learning_rate': 0.001
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "OptimizerTest" in optimized

        # Verify HPO expressions were replaced
        assert 'HPO(choice("Adam", "SGD", "RMSprop"))' not in optimized
        assert "HPO(log_range(1e-4, 1e-2))" not in optimized

        # Verify the optimizer was properly handled
        assert "optimizer: Adam(learning_rate=0.001)" in optimized

    def test_generate_optimized_dsl_with_special_characters(self):
        """Test HPO parameter replacement with special characters in strings."""
        config = """
        network SpecialCharsTest {
            input: (28, 28, 1)
            layers:
                Dense(128, activation=HPO(choice("relu", "leaky_relu", "elu")))
                Output(10, activation=HPO(choice("softmax", "sigmoid")))
            loss: HPO(choice("categorical_crossentropy", "binary_crossentropy"))
        }
        """

        best_params = {
            'dense_activation': 'leaky_relu',
            'output_activation': 'softmax',
            'loss': 'categorical_crossentropy'
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "SpecialCharsTest" in optimized

        # Verify HPO expressions were replaced
        assert 'HPO(choice("relu", "leaky_relu", "elu"))' not in optimized
        assert 'HPO(choice("softmax", "sigmoid"))' not in optimized
        assert 'HPO(choice("categorical_crossentropy", "binary_crossentropy"))' not in optimized

        # Verify the string values were properly handled (with quotes)
        assert 'activation="leaky_relu"' in optimized
        assert 'activation="softmax"' in optimized
        assert 'loss: "categorical_crossentropy"' in optimized

    def test_generate_optimized_dsl_with_numeric_edge_cases(self):
        """Test HPO parameter replacement with numeric edge cases (very small/large values)."""
        config = """
        network NumericEdgeCasesTest {
            input: (28, 28, 1)
            layers:
                Dense(128)
                Dropout(HPO(range(0.01, 0.99, step=0.01)))
                Output(10)
            optimizer: Adam(learning_rate=HPO(log_range(1e-10, 1.0)))
        }
        """

        best_params = {
            'dropout_rate': 0.01,  # Edge case: very small value
            'learning_rate': 1e-10  # Edge case: very small value
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "NumericEdgeCasesTest" in optimized

        # Verify HPO expressions were replaced
        assert "HPO(range(0.01, 0.99, step=0.01))" not in optimized
        assert "HPO(log_range(1e-10, 1.0))" not in optimized

        # Verify the edge case values were properly handled
        assert "Dropout(0.01)" in optimized
        assert "learning_rate=1e-10" in optimized

    def test_generate_optimized_dsl_with_multiple_hpo_in_same_line(self):
        """Test HPO parameter replacement when multiple HPO expressions are on the same line."""
        config = """
        network MultiHPOLineTest {
            input: (28, 28, 1)
            layers:
                Conv2D(HPO(choice(32, 64)), kernel_size=(HPO(choice(3, 5)), HPO(choice(3, 5))))
                MaxPooling2D(pool_size=(2, 2))
                Flatten()
                Dense(128)
                Output(10)
            optimizer: Adam(learning_rate=0.001)
        }
        """

        best_params = {
            'conv2d_filters': 64,
            'conv2d_kernel_size_0': 3,
            'conv2d_kernel_size_1': 5
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "MultiHPOLineTest" in optimized

        # Verify HPO expressions were replaced
        assert "HPO(choice(32, 64))" not in optimized
        assert "HPO(choice(3, 5))" not in optimized

        # Verify the values were properly handled
        assert "Conv2D(64, kernel_size=(3, 5)" in optimized

    def test_generate_optimized_dsl_with_boolean_values(self):
        """Test HPO parameter replacement with boolean values."""
        config = """
        network BooleanTest {
            input: (28, 28, 1)
            layers:
                BatchNormalization(center=HPO(choice(true, false)))
                Conv2D(32, kernel_size=(3, 3), use_bias=HPO(choice(true, false)))
                MaxPooling2D(pool_size=(2, 2))
                Flatten()
                Dense(128)
                Output(10)
            optimizer: Adam(learning_rate=0.001)
        }
        """

        best_params = {
            'batchnormalization_center': True,
            'conv2d_use_bias': False
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "BooleanTest" in optimized

        # Verify HPO expressions were replaced
        assert "HPO(choice(true, false))" not in optimized

        # Verify the boolean values were properly handled
        assert "center=true" in optimized or "center=True" in optimized
        assert "use_bias=false" in optimized or "use_bias=False" in optimized

    def test_generate_optimized_dsl_with_whitespace_variations(self):
        """Test HPO parameter replacement with different whitespace variations."""
        config = """
        network WhitespaceTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64,128,256)))
                Dropout(HPO(  range(  0.3  ,  0.7  ,  step=0.1  )  ))
                Output(10)
            optimizer: Adam(learning_rate=HPO(log_range(1e-4,1e-2)))
        }
        """

        best_params = {
            'dense_units': 128,
            'dropout_rate': 0.5,
            'learning_rate': 0.001
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "WhitespaceTest" in optimized

        # Verify HPO expressions were replaced
        assert "HPO(choice(64,128,256))" not in optimized
        assert "HPO(  range(  0.3  ,  0.7  ,  step=0.1  )  )" not in optimized
        assert "HPO(log_range(1e-4,1e-2))" not in optimized

        # Verify the values were properly handled
        assert "Dense(128)" in optimized
        assert "Dropout(0.5)" in optimized
        assert "learning_rate=0.001" in optimized

    def test_generate_optimized_dsl_with_comments(self):
        """Test HPO parameter replacement in a file with comments."""
        config = """
        # This is a test network with comments
        network CommentTest {
            input: (28, 28, 1)  # Input shape for MNIST
            layers:
                # First layer with HPO
                Dense(HPO(choice(64, 128, 256)))  # Try different sizes
                # Dropout layer with HPO
                Dropout(HPO(range(0.3, 0.7, step=0.1)))  # Try different dropout rates
                Output(10)  # 10 classes for MNIST
            # Use Adam optimizer with HPO learning rate
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """

        best_params = {
            'dense_units': 128,
            'dropout_rate': 0.5,
            'learning_rate': 0.001
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "CommentTest" in optimized

        # Verify HPO expressions were replaced
        assert "HPO(choice(64, 128, 256))" not in optimized
        assert "HPO(range(0.3, 0.7, step=0.1))" not in optimized
        assert "HPO(log_range(1e-4, 1e-2))" not in optimized

        # Verify the values were properly handled
        assert "Dense(128)" in optimized
        assert "Dropout(0.5)" in optimized
        assert "learning_rate=0.001" in optimized

        # Verify comments are preserved
        assert "# This is a test network with comments" in optimized
        assert "# Input shape for MNIST" in optimized
        assert "# Try different sizes" in optimized
        assert "# Try different dropout rates" in optimized
        assert "# 10 classes for MNIST" in optimized
