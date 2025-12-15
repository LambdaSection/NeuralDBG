# Neural DSL Test Suite - Bug Report and Prioritized Fix List

**Generated:** 2024
**Test Run:** Full test suite with pytest -v --tb=short
**Total Tests Collected:** 1024 tests
**Collection Errors:** 13 modules
**Test Failures:** 132 tests
**Test Passes:** 357 tests
**Skipped:** 11 tests

---

## Executive Summary

The test suite reveals several critical issues across multiple modules:

1. **Import/Dependency Errors (13 modules)** - Missing keras module and incorrect import paths
2. **Code Generation Errors (45 tests)** - Auto-flatten policy issues with Output layer
3. **Parser Edge Cases (11 tests)** - Device specification, execution config, HPO parsing issues
4. **Shape Propagation Errors (19 tests)** - Error handling and validation exceptions
5. **CLI Tests (13 tests)** - Missing 'name' attribute in CLI module
6. **File Operation Tests (3 tests)** - Directory creation issues before file write

---

## Category 1: CRITICAL - Import/Collection Errors (Priority: P0)

### Issue: Missing keras Module
**Impact:** Blocks 13 test modules from running
**Affected Modules:**
- tests/hpo/test_hpo_integration.py
- tests/integration_tests/test_complete_workflow_integration.py
- tests/integration_tests/test_edge_cases_workflow.py
- tests/integration_tests/test_end_to_end_scenarios.py
- tests/integration_tests/test_hpo_tracking_workflow.py

**Error:**
```python
neural/hpo/hpo.py:6: in <module>
    import keras
E   ModuleNotFoundError: No module named 'keras'
```

**Root Cause:** 
- keras is imported directly but not available in Python 3.14
- Should use tensorflow.keras instead

**Recommended Fix:**
```python
# neural/hpo/hpo.py line 6
# OLD: import keras
# NEW: from tensorflow import keras
```

### Issue: Incorrect Import Path in Dynamic Visualizer
**Affected:** tests/visualization/test_dynamic_visualizer.py

**Error:**
```python
neural/visualization/dynamic_visualizer/api.py:4: in <module>
    from parser.parser import create_parser, ModelTransformer
E   ModuleNotFoundError: No module named 'parser.parser'
```

**Root Cause:** 
- Using relative import 'parser.parser' instead of 'neural.parser.parser'

**Recommended Fix:**
```python
# neural/visualization/dynamic_visualizer/api.py line 4
# OLD: from parser.parser import create_parser, ModelTransformer
# NEW: from neural.parser.parser import create_parser, ModelTransformer
```

### Issue: Missing Class Exports
**Affected Modules:**
- tests/test_automl_coverage.py - Cannot import RandomSearch, BayesianSearch, EvolutionarySearch
- tests/test_cost.py - Cannot import ResourceMetrics
- tests/test_data_coverage.py - Cannot import QualityValidator
- tests/test_federated_coverage.py - Cannot import FederatedAveraging, SecureAggregation
- tests/test_mlops_coverage.py - Cannot import ModelDeployment
- tests/test_monitoring_coverage.py - Cannot import DataQualityChecker
- tests/tracking/test_experiment_tracker.py - Cannot import ArtifactVersion

**Root Cause:** These classes are not defined/exported in their respective modules

---

## Category 2: HIGH - Code Generation Output Layer Policy (Priority: P1)

### Issue: Output Layer Requires 2D Input Without auto_flatten_output
**Impact:** 45 failing tests across code generation suite
**Pattern:** 
```
neural.exceptions.CodeGenException: [ERROR] Layer 'Output' expects 2D input 
(batch, features) but got higher-rank. Insert a Flatten/GAP before it or pass 
auto_flatten_output=True.
```

**Affected Tests:**
- test_generate_tensorflow_complex
- test_generate_pytorch_complex
- test_tensorflow_layer_generation[Dense-params4-...]
- test_pytorch_layer_generation[Dense-params4-...]
- test_invalid_activation_handling
- test_shape_propagation
- test_custom_optimizer_params
- test_transformer_generation (and 6 related tests)
- test_different_pooling_configs
- test_rnn_types_and_configs
- test_batch_norm_params
- test_mixed_precision_training
- test_model_saving_loading
- test_ensure_2d_*_higher_rank_no_auto_flatten (2 tests)
- test_training_config_with_mixed_precision
- test_output_requires_flatten_policy[False-tensorflow]
- test_output_requires_flatten_policy[False-pytorch]

**Root Cause:**
Tests create models with conv/pooling layers followed by Output layer without Flatten, but auto_flatten_output policy is not enabled in test fixtures.

**Recommended Fix Options:**
1. Update test fixtures to include `auto_flatten_output=True` in model_data
2. Add Flatten layer before Output in test models
3. Relax Output layer policy to auto-insert Flatten when disabled

**Example Fix for Tests:**
```python
# Add to model_data dict:
model_data = {
    'input': ...,
    'layers': [...],
    'auto_flatten_output': True  # Add this
}
```

---

## Category 3: MEDIUM - Parser Edge Cases (Priority: P2)

### Issue: Missing Device Specification in Parsed Output
**Affected Tests:**
- test_valid_device_cpu
- test_valid_device_cuda_with_index
- test_valid_device_tpu

**Error:**
```python
KeyError: 'device'
```

**Root Cause:** Parser doesn't preserve device specification in transformed output

### Issue: Missing execution_config in Parsed Network
**Affected:** test_network_with_execution_config

**Error:**
```python
AssertionError: assert 'execution_config' in {...}
```

**Root Cause:** Parser doesn't extract/preserve execution_config section

### Issue: Missing Optimizer/Loss Validation
**Affected Tests:**
- test_network_missing_optimizer
- test_network_missing_loss

**Error:**
```python
KeyError: 'optimizer'
KeyError: 'loss'
```

**Root Cause:** Parser should validate required fields but doesn't enforce them

### Issue: HPO Parsing Edge Cases
**Affected Tests:**
- test_multiple_hpo_expressions_in_single_param
- test_layer_with_empty_sublayers  
- test_split_params_empty_string
- test_network_with_multiple_inputs
- test_network_with_training_config_zero_epochs

**Root Cause:** Edge case handling in HPO expression parsing and validation

---

## Category 4: MEDIUM - Shape Propagation Errors (Priority: P2)

### Issue: Error Handling Uses Exceptions Instead of Returning Error Objects
**Impact:** 19 tests expect errors to be caught/handled gracefully

**Affected Tests:**
- test_conv2d_kernel_too_large (2 instances)
- test_dense_with_4d_input
- test_conv2d_missing_filters
- test_dense_missing_units
- test_conv2d_negative_filters
- test_maxpooling2d_pool_size_too_large
- test_conv2d_negative_stride
- test_maxpooling2d_negative_stride
- test_invalid_input_shape_empty
- test_invalid_input_shape_negative
- test_missing_layer_type
- test_missing_params
- test_propagate_missing_type_key
- test_propagate_empty_input_shape
- test_propagate_negative_input_dimensions
- test_conv2d_kernel_exceeds_input
- test_conv2d_invalid_output_dimensions
- test_dense_higher_dimensional_input (2 instances)
- test_validate_conv_invalid_dimensions
- test_validate_conv_kernel_too_large
- test_validate_dense_higher_d_input

**Errors Raised:**
- neural.exceptions.ShapeMismatchError
- neural.exceptions.InvalidParameterError
- neural.exceptions.InvalidShapeError

**Root Cause:** Tests may expect these to be warnings or handled errors, not exceptions. Or tests need to use pytest.raises().

### Issue: Activation Anomaly Detection Assertion
**Affected:** test_detect_activation_anomalies_with_torch

**Error:**
```python
assert False is True
```

**Root Cause:** Detection function returns False when it should return True, or vice versa

---

## Category 5: MEDIUM - CLI Tests (Priority: P2)

### Issue: Missing 'name' Attribute on CLI Module
**Impact:** 13 CLI tests fail
**Affected Tests:**
- test_clean_removes_generated_artifacts
- test_clean_no_items
- test_compile_command
- test_compile_pytorch_backend
- test_compile_dry_run
- test_compile_invalid_file
- test_compile_invalid_backend
- test_compile_invalid_syntax
- test_run_command
- test_run_invalid_file
- test_visualize_invalid_file
- test_version_command
- test_debug_invalid_file

**Error:**
```python
AttributeError: module 'neural.cli.cli' has no attribute 'name'
```

**Root Cause:** CLI tests trying to access `cli.name` attribute that doesn't exist. Tests may be expecting Click command group.

**Recommended Fix:** 
Check if tests should use `cli.cli` (the command group function) instead of expecting a `name` attribute on the module.

---

## Category 6: LOW - File Operation Tests (Priority: P3)

### Issue: Directory Not Created Before File Write
**Affected Tests:**
- test_save_file_invalid_path
- test_file_handling_errors

**Error:**
```python
neural.exceptions.FileOperationError: [ERROR] File write failed: ...
Reason: [Errno 2] No such file or directory
```

**Root Cause:** Code generator doesn't create parent directories before writing files

**Recommended Fix:**
```python
# Add before file write:
os.makedirs(os.path.dirname(filepath), exist_ok=True)
```

### Issue: File Type Validation
**Affected:** test_load_file_unsupported_extension

**Error:**
```python
neural.exceptions.FileOperationError: [ERROR] File parse failed: ...test.txt. 
Reason: Unsupported file type. Expected .neural, .nr, or .rnr
```

**Expected:** Test expects this error (correct behavior)

### Issue: Missing File Error
**Affected:** test_load_file_nonexistent

**Error:**
```python
neural.exceptions.FileOperationError: [ERROR] File read failed: nonexistent.neural
```

**Expected:** Test expects this error (correct behavior)

---

## Category 7: LOW - Model Data Validation (Priority: P3)

### Issue: Model Data Validation Exceptions
**Impact:** 6 tests validate error handling (working as intended)

**Affected Tests:**
- test_generate_code_invalid_model_data_type
- test_generate_code_missing_layers_key
- test_generate_code_missing_input_key
- test_generate_code_invalid_layer_format
- test_generate_code_layer_missing_type
- test_generate_code_unsupported_backend

**Status:** Tests are passing - they expect and receive appropriate exceptions

---

## Category 8: LOW - Layer Multiplication Validation (Priority: P3)

### Issue: Layer Multiplication Parameter Validation
**Impact:** 3 tests validate error handling (working as intended)

**Affected Tests:**
- test_multiply_value_zero
- test_multiply_value_negative
- test_multiply_value_non_integer

**Status:** Tests expect InvalidParameterError and receive it correctly

---

## Category 9: LOW - Layer Generation Issues (Priority: P3)

### Issue: Layer Not Generated in Code
**Affected Tests:**
- test_generate_pytorch_channels_first (Conv2D not in output)
- test_tensorflow_layer_generation[Conv2D-...] (Conv2D not matching expected)
- test_pytorch_layer_generation[Conv2D-...] (Conv2D not in output)
- test_pytorch_layer_generation[LSTM-...] (LSTM not in output)
- test_pytorch_layer_generation[BatchNormalization-...] (BatchNorm not in output)

**Root Cause:** Layers are not being generated in output code, possibly due to:
1. Shape propagation failing before layer generation
2. Layer filtering based on validation errors
3. Missing layer implementation

### Issue: Custom Layer Warning Not Emitted
**Affected:** test_custom_layer_handling

**Error:**
```python
Failed: DID NOT WARN. No warnings of type (<class 'UserWarning'>,) were emitted.
```

**Root Cause:** Custom layer handling doesn't emit expected warning

### Issue: ONNX Model Structure Validation
**Affected:** test_onnx_model_structure

**Error:**
```python
onnx.onnx_cpp2py_export.checker.ValidationError: 
Field 'shape' of 'type' is required but missing.
```

**Root Cause:** ONNX model generation creates invalid ONNX proto (missing shape field)

---

## Prioritized Fix List

### Phase 1: Critical Blockers (P0) - Must Fix First
1. **Fix keras import** in neural/hpo/hpo.py
   - Change `import keras` to `from tensorflow import keras`
   - Estimated effort: 5 minutes
   - Impact: Unblocks 5 test modules

2. **Fix parser import** in neural/visualization/dynamic_visualizer/api.py
   - Change `from parser.parser` to `from neural.parser.parser`
   - Estimated effort: 2 minutes
   - Impact: Unblocks 1 test module

3. **Implement missing classes** in various modules
   - RandomSearch, BayesianSearch, EvolutionarySearch in automl/search_strategies.py
   - ResourceMetrics in cost/__init__.py
   - QualityValidator in data/quality_validator.py
   - FederatedAveraging, SecureAggregation in federated/aggregation.py
   - ModelDeployment in mlops/deployment.py
   - DataQualityChecker in monitoring/data_quality.py
   - ArtifactVersion in tracking/__init__.py
   - Estimated effort: 2-4 hours (depends on implementation complexity)
   - Impact: Unblocks 7 test modules

### Phase 2: High Priority (P1) - Fix Soon
4. **Fix Output layer auto-flatten policy** in code generation
   - Option A: Update all test fixtures to include `auto_flatten_output=True`
   - Option B: Modify policy to auto-insert Flatten when needed
   - Estimated effort: 2-3 hours
   - Impact: Fixes 45 failing tests

### Phase 3: Medium Priority (P2) - Fix Next Sprint
5. **Parser device specification preservation**
   - Ensure device specs are extracted and preserved in AST
   - Estimated effort: 1 hour
   - Impact: Fixes 3 tests

6. **Parser execution_config support**
   - Add execution_config section parsing
   - Estimated effort: 1 hour
   - Impact: Fixes 1 test

7. **Parser validation for required fields**
   - Validate optimizer and loss are present
   - Estimated effort: 30 minutes
   - Impact: Fixes 2 tests

8. **Parser HPO edge cases**
   - Handle multiple HPO expressions in single param
   - Handle empty sublayers
   - Handle empty strings in param split
   - Estimated effort: 2 hours
   - Impact: Fixes 5 tests

9. **Shape propagation error handling**
   - Review if tests need pytest.raises() or if errors should be warnings
   - Estimated effort: 1-2 hours
   - Impact: Fixes 19 tests

10. **CLI module 'name' attribute**
    - Investigate what tests expect and fix CLI module structure
    - Estimated effort: 1 hour
    - Impact: Fixes 13 tests

### Phase 4: Low Priority (P3) - Nice to Have
11. **File operation directory creation**
    - Add os.makedirs before file writes
    - Estimated effort: 15 minutes
    - Impact: Fixes 2 tests

12. **Layer generation issues**
    - Debug why Conv2D, LSTM, BatchNorm not generated
    - Fix custom layer warning
    - Fix ONNX shape field
    - Estimated effort: 3-4 hours
    - Impact: Fixes 7 tests

---

## Test Success Rate by Module

| Module | Total | Passed | Failed | Skip | Success Rate |
|--------|-------|--------|--------|------|--------------|
| parser/ | 200+ | 189 | 11 | 0 | 94.5% |
| code_generator/ | 100+ | 59 | 45 | 0 | 56.7% |
| shape_propagation/ | 80+ | 61 | 19 | 0 | 76.3% |
| cli/ | 15+ | 2 | 13 | 0 | 13.3% |
| integration_tests/ | 40+ | 35 | 0 | 5 | 100% (excl. blocked) |
| dashboard/ | 20+ | 20 | 0 | 0 | 100% |
| hpo/ | 30+ | 30 | 0 | 0 | 100% |
| Other modules | 500+ | N/A | N/A | N/A | Not collected |

---

## Recommendations

1. **Immediate Action:** Fix all P0 issues to unblock test collection
2. **Code Review:** Review auto_flatten_output policy - decide if tests or implementation needs change
3. **Documentation:** Document expected behavior for edge cases in parser
4. **Test Coverage:** After fixes, run full suite again to identify remaining issues
5. **CI/CD:** Add test suite to CI pipeline to catch regressions early
6. **Deprecation:** Consider removing or stubbing federated/automl/mlops modules if they're marked for deprecation

---

## Additional Notes

- Test suite uses Python 3.14 which has deprecated some APIs (asyncio.iscoroutinefunction)
- dash_table.DataTable will be removed in future - consider migrating to dash-ag-grid
- Some tests may timeout on slower systems (observed 600s timeout on full suite)
- Consider splitting test suite into fast/slow categories for development workflow

---

**End of Report**
