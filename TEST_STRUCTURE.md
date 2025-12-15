# Neural DSL Test Structure

## Overview
This document describes the test suite structure after cleanup.

## Test Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── README.md                   # Test documentation
├── TEST_COVERAGE_SUMMARY.md    # Coverage metrics
│
├── parser/                     # Parser tests (CORE)
│   ├── test_basic.py
│   ├── test_parser.py
│   ├── test_parser_edge_cases.py
│   ├── test_layers.py
│   ├── test_networks.py
│   ├── test_properties.py
│   ├── test_research.py
│   ├── test_wrapper.py
│   ├── test_macro.py
│   ├── test_fuzzing.py
│   └── test_validation.py
│
├── shape_propagation/          # Shape propagation tests (CORE)
│   ├── test_basic_propagation.py
│   ├── test_complex_models.py
│   ├── test_enhanced_propagator.py
│   ├── test_error_handling.py
│   ├── test_shape_propagator_edge_cases.py
│   ├── test_hpo_shape_propagation.py
│   └── test_visualization.py
│
├── code_generator/             # Code generation tests (CORE)
│   ├── test_code_generator.py
│   ├── test_code_generator_edge_cases.py
│   ├── test_policy_and_parity.py
│   └── test_policy_helpers.py
│
├── cli/                        # CLI tests
│   ├── test_cli.py
│   └── test_clean_command.py
│
├── hpo/                        # Hyperparameter optimization tests
│   ├── test_hpo_cli_integration.py
│   ├── test_hpo_code_generation.py
│   ├── test_hpo_fix.py
│   ├── test_hpo_integration.py
│   ├── test_hpo_optimizers.py
│   └── test_hpo_parser.py
│
├── dashboard/                  # Dashboard tests
│   ├── test_dashboard.py
│   └── test_data_to_dashboard.py
│
├── visualization/              # Visualization tests
│   ├── test_cli_visualization.py
│   ├── test_dashboard_visualization.py
│   ├── test_dynamic_visualizer.py
│   ├── test_neural_visualizer.py
│   ├── test_shape_visualization.py
│   ├── test_static_visualizer.py
│   ├── test_tensor_flow.py
│   └── run_tests.py
│
├── integration_tests/          # Integration tests (SLOW)
│   ├── test_complete_workflow_integration.py
│   ├── test_device_integration.py
│   ├── test_edge_cases_workflow.py
│   ├── test_end_to_end_scenarios.py
│   ├── test_hpo_tracking_workflow.py
│   ├── test_onnx_workflow.py
│   ├── test_tracking_integration.py
│   ├── test_transformer_workflow.py
│   ├── Parsing_Compilation_Execution_Debugging_Dashboard.py
│   └── run_all_tests.py
│
├── cloud/                      # Cloud execution tests
│   ├── test_cloud_executor.py
│   ├── test_cloud_integration.py
│   ├── test_interactive_shell.py
│   └── test_notebook_interface.py
│
├── benchmarks/                 # Benchmark tests
│   ├── test_benchmarks.py
│   ├── test_benchmark_runner.py
│   ├── test_metrics_collector.py
│   ├── benchmark_runner.py
│   ├── benchmark_suite.py
│   ├── models.py
│   ├── run_benchmarks.py
│   ├── example_usage.py
│   └── verify_setup.py
│
├── performance/                # Performance tests
│   ├── test_cli_startup.py
│   ├── test_end_to_end.py
│   ├── test_parser_performance.py
│   ├── test_shape_propagation.py
│   ├── benchmark_runner.py
│   ├── profiling_utils.py
│   └── profile_cli.py
│
├── tracking/                   # Experiment tracking tests
│   └── test_experiment_tracker.py
│
├── utils/                      # Utility tests
│   └── test_seeding.py
│
├── aquarium_ide/               # Aquarium IDE tests (EXCLUDED by default)
│   ├── conftest.py
│   ├── test_backend_api.py
│   ├── test_e2e_workflows.py
│   ├── test_integration.py
│   ├── test_real_examples.py
│   └── test_welcome_components.py
│
├── aquarium_e2e/               # E2E tests with Playwright (EXCLUDED by default)
│   ├── conftest.py
│   ├── page_objects.py
│   ├── utils.py
│   ├── test_compilation.py
│   ├── test_complete_workflow.py
│   ├── test_data.py
│   ├── test_dsl_editor.py
│   ├── test_export.py
│   ├── test_model_variations.py
│   ├── test_navigation.py
│   ├── test_performance.py
│   ├── test_ui_elements.py
│   └── run_tests.py
│
├── docgen/                     # Documentation generation tests
│   └── test_docgen_v11.py
│
└── Individual test files:
    ├── test_automl.py          # AutoML tests
    ├── test_automl_coverage.py
    ├── test_codegen_and_docs.py
    ├── test_cost.py            # Cost analysis tests
    ├── test_cost_coverage.py
    ├── test_data_coverage.py   # Data management tests
    ├── test_data_versioning.py
    ├── test_debugger.py        # Debugger tests
    ├── test_device_execution.py
    ├── test_error_suggestions.py
    ├── test_examples.py        # Example validation
    ├── test_federated_coverage.py  # Federated learning tests
    ├── test_integrations.py    # Platform integrations tests
    ├── test_marketplace.py     # Marketplace tests
    ├── test_mlops_coverage.py  # MLOps tests
    ├── test_monitoring_coverage.py  # Monitoring tests
    ├── test_no_code_interface.py
    ├── test_pretrained.py      # Pretrained models tests
    ├── test_seed.py           # Random seed tests
    ├── test_shape_propagation.py
    ├── test_shape_propagation_fixes.py
    ├── test_teams.py          # Team management tests
    └── ui_tests.py           # UI tests
```

## Test Categories

### Core Tests (Always Run)
These test the fundamental DSL functionality:
- **Parser tests**: DSL syntax parsing and validation
- **Shape propagation tests**: Shape inference and validation
- **Code generation tests**: Multi-backend code generation

Run with:
```bash
python -m pytest tests/parser/ tests/shape_propagation/ tests/code_generator/ -v
```

### Unit Tests
Fast, isolated tests for individual components.

Run with:
```bash
python -m pytest tests/ -v -m "unit"
```

### Integration Tests
Tests that verify component interactions. May be slower.

Run with:
```bash
python -m pytest tests/ -v -m "integration"
```

### Slow Tests
Tests that take significant time (benchmarks, performance tests).

Skip with:
```bash
python -m pytest tests/ -v -m "not slow"
```

### E2E Tests (Excluded by Default)
Browser-based tests requiring Playwright. Must be explicitly included.

Run with:
```bash
# Install Playwright first
pip install playwright
playwright install

# Run E2E tests
python -m pytest tests/aquarium_e2e/ -v
python -m pytest tests/aquarium_ide/ -v
```

## Test Markers

Configured in `pyproject.toml` and `tests/conftest.py`:

- `unit`: Unit tests (fast, isolated)
- `integration`: Integration tests (component interactions)
- `slow`: Slow-running tests (skip for quick validation)
- `requires_gpu`: Tests requiring GPU/CUDA
- `requires_torch`: Tests requiring PyTorch
- `requires_tensorflow`: Tests requiring TensorFlow
- `requires_onnx`: Tests requiring ONNX
- `parser`: Parser-related tests
- `codegen`: Code generation tests
- `shape`: Shape propagation tests
- `hpo`: HPO tests
- `dashboard`: Dashboard tests
- `cloud`: Cloud execution tests
- `aquarium`: Aquarium IDE tests
- `e2e`: End-to-end tests

## Running Tests

### Quick Validation
```bash
# Check imports
python check_imports.py

# Collect tests only (verify no import errors)
python -m pytest tests/ --collect-only -q

# Run core tests
python -m pytest tests/parser/ tests/shape_propagation/ tests/code_generator/ -v
```

### Full Test Suite
```bash
# Run comprehensive test suite with reporting
python run_tests_after_cleanup.py

# OR run manually
python -m pytest tests/ -v --tb=short

# Skip slow tests
python -m pytest tests/ -v -m "not slow"

# Only unit tests
python -m pytest tests/ -v -m "unit"
```

### Specific Test Groups
```bash
# Parser tests only
python -m pytest tests/parser/ -v

# With coverage
python -m pytest tests/parser/ -v --cov=neural.parser --cov-report=term
```

### Backend-Specific Tests
```bash
# TensorFlow tests
python -m pytest tests/ -v -m "requires_tensorflow"

# PyTorch tests
python -m pytest tests/ -v -m "requires_torch"

# ONNX tests
python -m pytest tests/ -v -m "requires_onnx"
```

## Test Dependencies

### Core Dependencies (Required)
- pytest
- numpy
- lark

### Optional Dependencies
- torch (for PyTorch backend tests)
- tensorflow (for TensorFlow backend tests)
- onnx (for ONNX backend tests)
- plotly (for visualization tests)
- graphviz (for graph visualization tests)
- dash (for dashboard tests)
- playwright (for E2E tests)
- optuna (for HPO tests)
- scikit-learn (for AutoML tests)

## Excluded Directories

Configured in `pyproject.toml`:
- `tests/aquarium_e2e/` - Requires Playwright
- `tests/aquarium_ide/` - Requires special setup
- `tests/tmp_path/` - Temporary files

These are excluded from default test collection to avoid dependency issues.

## Test Fixtures

Key fixtures defined in `tests/conftest.py`:

### Parser Fixtures
- `parser()`: Default parser
- `layer_parser()`: Layer-specific parser
- `network_parser()`: Network-specific parser
- `research_parser()`: Research notation parser
- `transformer()`: ModelTransformer instance

### Sample DSL Fixtures
- `sample_dsl_simple()`: Simple network
- `sample_dsl_cnn()`: CNN example
- `sample_dsl_rnn()`: RNN example
- `sample_dsl_transformer()`: Transformer example
- `sample_dsl_hpo()`: HPO example

### Configuration Fixtures
- `sample_shapes()`: Common input shapes
- `sample_layer_configs()`: Layer configurations
- `sample_optimizer_configs()`: Optimizer configurations
- `sample_loss_functions()`: Loss function names
- `backend()`: Parametrized backend fixture (tf/pytorch/onnx)

## Coverage

Generate coverage report:
```bash
# Run tests with coverage
python -m pytest tests/ -v --cov=neural --cov-report=term --cov-report=html

# Generate summary
python generate_test_coverage_summary.py
```

Coverage reports:
- Terminal: Printed to console
- HTML: `htmlcov/index.html`
- Summary: `TEST_COVERAGE_SUMMARY.md`
