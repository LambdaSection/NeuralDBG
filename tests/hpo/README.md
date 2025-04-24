# Hyperparameter Optimization (HPO) Tests

## Overview

This directory contains tests for the hyperparameter optimization (HPO) functionality in the Neural framework. The HPO system allows users to define searchable hyperparameter spaces in their Neural DSL models and automatically find optimal configurations.

## Test Files

### test_hpo_integration.py

This file tests the integration of the HPO system with other components of the Neural framework. It includes:

1. **Forward Pass Tests**
   - Tests that models with HPO parameters can correctly process input data
   - Verifies output shapes for both flat and convolutional inputs
   - Ensures proper tensor format conversion between NHWC and NCHW

2. **Training Loop Tests**
   - Tests the training functionality with mock data loaders
   - Verifies convergence behavior with simple models
   - Tests error handling for invalid optimizer configurations

3. **HPO Objective Tests**
   - Tests the objective function used for hyperparameter optimization
   - Verifies multi-objective optimization metrics (loss, accuracy, precision, recall)
   - Tests handling of different HPO parameter types in the objective function

4. **Parser Tests**
   - Tests parsing of HPO configurations in the Neural DSL
   - Verifies handling of different HPO types (categorical, range, log_range)
   - Tests validation and error handling for invalid configurations

5. **HPO Integration Tests**
   - Tests the full HPO pipeline from configuration to optimized model
   - Verifies that optimized models can be generated and executed
   - Tests the code generation for optimized models

6. **Edge Cases and Error Handling**
   - Tests behavior with invalid parameters
   - Tests models with minimal layer configurations
   - Verifies proper error messages for invalid inputs

## Mock Objects

The tests use several mock objects to isolate the HPO components:

- **MockTrial**: Simulates an Optuna trial object for hyperparameter suggestion
- **mock_data_loader**: Creates synthetic data for training and validation

## Running the Tests

To run all HPO tests:

```bash
python -m pytest tests/hpo
```

To run a specific test file:

```bash
python -m pytest tests/hpo/test_hpo_integration.py
```

To run a specific test function:

```bash
python -m pytest tests/hpo/test_hpo_integration.py::test_hpo_integration_full_pipeline
```

## Dependencies

These tests depend on:

- PyTorch for model creation and training
- Optuna (implicitly) for the hyperparameter optimization framework
- The Neural DSL parser and transformer
- The Neural code generation system

## Integration with Other Components

The HPO tests verify integration with:

1. **Parser**: Tests that HPO syntax in the DSL is correctly parsed
2. **Model Creation**: Tests that models with HPO parameters can be created
3. **Training**: Tests that models with HPO parameters can be trained
4. **Code Generation**: Tests that optimized models can be generated from HPO results
