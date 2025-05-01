# Shape Propagation Tests

This folder contains tests for the Neural Shape Propagation module, which is responsible for inferring and validating tensor shapes throughout a neural network model.

## Test Structure

The tests are organized into the following files:

1. **test_basic_propagation.py**: Tests for basic shape propagation functionality with different layer types and configurations.
2. **test_hpo_shape_propagation.py**: Tests for handling Hyperparameter Optimization (HPO) parameters in shape propagation.
3. **test_complex_models.py**: Tests for shape propagation in complex model architectures with multiple layers and branches.
4. **test_error_handling.py**: Tests for error handling and validation in shape propagation.
5. **test_visualization.py**: Tests for shape visualization functionality.

## Running the Tests

You can run all shape propagation tests with:

```bash
pytest tests/shape_propagation
```

Or run a specific test file:

```bash
pytest tests/shape_propagation/test_basic_propagation.py
```

## Test Coverage

These tests cover:

- Shape propagation for all supported layer types
- Handling of different data formats (channels_first, channels_last)
- Handling of HPO parameters in shape propagation
- Error detection and validation
- Shape visualization and reporting
- Complex model architectures with multiple branches and connections
