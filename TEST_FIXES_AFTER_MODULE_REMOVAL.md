# Test Fixes After Module Removal

## Summary

Fixed broken tests after module removals and import fixes. All 11 collection errors were resolved.

## Issues Fixed

### 1. Syntax Error in `neural/hpo/hpo.py` (Line 631)
**Problem**: Corrupted/duplicate content at the end of the file caused unterminated string literal error.

**Solution**: Removed duplicate lines (631-634) that contained corrupted text:
```python
# REMOVED:
# ate'] = 0.001  # Default from optimizer_config
# 
#     return normalized_params
# ms
```

### 2. Missing HPO Validation Functions
**Problem**: `validate_hpo_categorical` and `validate_hpo_bounds` were called but not defined.

**Solution**: Added validation helper functions to `neural/hpo/hpo.py`:
```python
def validate_hpo_categorical(param_name: str, values: List[Any]) -> List[Any]:
    if not isinstance(values, list) or len(values) == 0:
        raise InvalidHPOConfigError(...)
    return values

def validate_hpo_bounds(param_name: str, low: float, high: float, hpo_type: str) -> Tuple[float, float]:
    if low is None or high is None:
        raise InvalidHPOConfigError(...)
    if low >= high:
        raise InvalidHPOConfigError(...)
    return low, high
```

### 3. TensorFlow Import Error for `DynamicTFModel`
**Problem**: `DynamicTFModel` class tried to inherit from `tf.keras.Model` when TensorFlow wasn't installed.

**Solution**: Wrapped the class definition in a conditional check:
```python
if HAS_TENSORFLOW:
    class DynamicTFModel(tf.keras.Model):
        # ... implementation
else:
    DynamicTFModel = None
```

### 4. Missing Classes in `neural/benchmarks/metrics_collector.py`
**Problem**: Test file imported `MemoryProfiler` and `ThroughputMeter` classes that didn't exist.

**Solution**: Added missing classes:
- `MemoryProfiler`: Profiles memory usage during execution
- `ThroughputMeter`: Measures throughput (samples/second)
- Enhanced `MetricsCollector` with missing methods:
  - `start_collection()`
  - `collect_snapshot()`
  - `get_system_info()`
- Enhanced `PerformanceTimer` with:
  - `duration` property
  - `get_duration_ms()` method

### 5. Missing Modules: `docgen`, `cost`, `monitoring`
**Problem**: Tests tried to import from non-existent modules.

**Solution**: Deleted test files for removed modules:
- `tests/test_cost.py`
- `tests/test_cost_coverage.py`
- `tests/test_monitoring_coverage.py`
- `tests/test_codegen_and_docs.py`
- `tests/docgen/test_docgen_v11.py`

### 6. Missing Classes in `neural/hpo/__init__.py`
**Problem**: `__init__.py` tried to import non-existent classes from `hpo.py`:
- `MultiObjectiveOptimizer`
- `DistributedHPO`
- `BayesianParameterImportance`

**Solution**: Updated imports to only include classes that exist:
```python
from .hpo import (
    optimize_and_return,
    objective,
    train_model,
    create_dynamic_model,
    get_data,
    resolve_hpo_params,
    DynamicPTModel,
)
```

### 7. Missing Dependencies: `seaborn`
**Problem**: `neural/hpo/parameter_importance.py` and `neural/hpo/visualization.py` required seaborn unconditionally.

**Solution**: Made seaborn and sklearn imports optional:

**parameter_importance.py**:
```python
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    RandomForestRegressor = None
    LabelEncoder = None
    StandardScaler = None
```

**visualization.py**:
```python
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

# Use fallback to matplotlib when seaborn not available
if HAS_SEABORN:
    sns.boxplot(x=param, y=metric, data=df, ax=ax)
else:
    # Fallback to scatter plot
    ax.scatter(x_indices, y, alpha=0.6)
```

## Test Results

### Before Fixes
- **Collection Errors**: 11 errors
- **Test Files Affected**: 
  - `tests/benchmarks/test_metrics_collector.py`
  - `tests/docgen/test_docgen_v11.py`
  - `tests/hpo/test_hpo_integration.py`
  - `tests/integration_tests/test_complete_workflow_integration.py`
  - `tests/integration_tests/test_edge_cases_workflow.py`
  - `tests/integration_tests/test_end_to_end_scenarios.py`
  - `tests/integration_tests/test_hpo_tracking_workflow.py`
  - `tests/test_codegen_and_docs.py`
  - `tests/test_cost.py`
  - `tests/test_cost_coverage.py`
  - `tests/test_monitoring_coverage.py`

### After Fixes
- **Collection Errors**: 0
- **Tests Collected Successfully**:
  - `tests/benchmarks/test_metrics_collector.py`: 13 tests (all passed)
  - `tests/test_error_suggestions.py`: 34 tests (all passed)
  - `tests/hpo/`: 79 tests collected successfully
- **Tests Deleted**: 5 test files for removed modules

## Verification

### Successful Test Runs
1. **Metrics Collector Tests**:
   ```bash
   pytest tests/benchmarks/test_metrics_collector.py -v
   # Result: 13 passed, 1 warning in 10.93s
   ```

2. **Error Suggestions Tests**:
   ```bash
   pytest tests/test_error_suggestions.py -v
   # Result: 34 passed, 2 warnings in 9.56s
   ```

3. **HPO Tests Collection**:
   ```bash
   pytest tests/hpo/ -v --collect-only
   # Result: 79 tests collected in 7.82s
   ```

## Files Modified

1. `neural/hpo/hpo.py`: Fixed syntax error, added validation functions, wrapped TF class
2. `neural/benchmarks/metrics_collector.py`: Added missing classes and methods
3. `neural/hpo/__init__.py`: Updated imports to match available classes
4. `neural/hpo/parameter_importance.py`: Made sklearn and seaborn optional
5. `neural/hpo/visualization.py`: Made seaborn optional with matplotlib fallback

## Files Deleted

1. `tests/test_cost.py`
2. `tests/test_cost_coverage.py`
3. `tests/test_monitoring_coverage.py`
4. `tests/test_codegen_and_docs.py`
5. `tests/docgen/test_docgen_v11.py`

## Notes

- All import errors have been resolved
- Optional dependencies (seaborn, sklearn) are now properly handled
- Tests can now be collected and run without import errors
- The full test suite may take time to run due to HPO tests requiring model training
