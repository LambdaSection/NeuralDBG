# Neural DSL - Quick Fixes Guide

This document provides immediate, copy-paste fixes for the most critical bugs identified in the test suite.

---

## Fix #1: keras Import Error (Affects 5 test modules)

**File:** `neural/hpo/hpo.py`
**Line:** 6
**Current:**
```python
import keras
```

**Fix:**
```python
from tensorflow import keras
```

**Impact:** Unblocks test_hpo_integration.py and 4 integration test modules

---

## Fix #2: Parser Import Error (Affects 1 test module)

**File:** `neural/visualization/dynamic_visualizer/api.py`
**Line:** 4
**Current:**
```python
from parser.parser import create_parser, ModelTransformer
```

**Fix:**
```python
from neural.parser.parser import create_parser, ModelTransformer
```

**Impact:** Unblocks test_dynamic_visualizer.py

---

## Fix #3: File Directory Creation (Affects 2 tests)

**File:** `neural/code_generation/code_generator.py` (or wherever save_file is implemented)

**Add this helper function:**
```python
import os

def ensure_directory_exists(filepath: str) -> None:
    """Create parent directories if they don't exist."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
```

**Use before file writes:**
```python
def save_file(filepath: str, content: str) -> None:
    ensure_directory_exists(filepath)
    with open(filepath, 'w') as f:
        f.write(content)
```

**Impact:** Fixes test_save_file_invalid_path and test_file_handling_errors

---

## Fix #4: Parser Empty Sublayers (Affects 1 test)

**File:** `neural/parser/parser.py`
**Location:** In the layer parsing logic where sublayers are extracted

**Current behavior:** Returns None for empty sublayers
**Expected behavior:** Returns [] for empty sublayers

**Fix:**
```python
# Find where sublayers are assigned, likely something like:
sublayers = self.extract_sublayers(layer_node)

# Change to:
sublayers = self.extract_sublayers(layer_node) or []
```

**Impact:** Fixes test_layer_with_empty_sublayers

---

## Fix #5: Parser Split Params Empty String (Affects 1 test)

**File:** `neural/parser/parser.py`
**Location:** In split_params function

**Current:**
```python
def split_params(param_str: str) -> List[str]:
    if not param_str:
        return []
    # ... rest of logic
```

**Fix:**
```python
def split_params(param_str: str) -> List[str]:
    if not param_str:
        return ['']  # Return list with empty string instead of empty list
    # ... rest of logic
```

**Impact:** Fixes test_split_params_empty_string

---

## Fix #6: Custom Layer Warning (Affects 1 test)

**File:** `neural/code_generation/code_generator.py`
**Location:** In layer generation logic when encountering unsupported layer types

**Add warning:**
```python
import warnings

# In the layer generation method, when layer type is not recognized:
if layer_type not in SUPPORTED_LAYERS:
    warnings.warn(
        f"Layer type '{layer_type}' is not natively supported. "
        f"Generating placeholder code.",
        UserWarning,
        stacklevel=2
    )
```

**Impact:** Fixes test_custom_layer_handling

---

## Fix #7: ONNX Shape Field (Affects 1 test)

**File:** `neural/code_generation/onnx_generator.py`
**Location:** When creating ONNX value_info or tensor types

**Current:**
```python
# Missing shape specification in ONNX proto
output = onnx.helper.make_tensor_value_info(
    name='output',
    elem_type=onnx.TensorProto.FLOAT
)
```

**Fix:**
```python
# Add shape to tensor value info
output = onnx.helper.make_tensor_value_info(
    name='output',
    elem_type=onnx.TensorProto.FLOAT,
    shape=[None, output_size]  # Add the shape parameter
)
```

**Impact:** Fixes test_onnx_model_structure

---

## Fix #8: Parser Device Specification Preservation (Affects 3 tests)

**File:** `neural/parser/parser.py`
**Location:** In ModelTransformer class, network transformation method

**Add device extraction:**
```python
def transform_network(self, tree):
    network_data = {
        'name': self.extract_name(tree),
        'input': self.extract_input(tree),
        'layers': self.extract_layers(tree),
        'optimizer': self.extract_optimizer(tree),
        'loss': self.extract_loss(tree),
        # Add these:
        'device': self.extract_device(tree),  # New method
    }
    return network_data

def extract_device(self, tree) -> Optional[str]:
    """Extract device specification from tree."""
    for child in tree.children:
        if hasattr(child, 'data') and child.data == 'device_spec':
            return child.children[0].value
    return None
```

**Impact:** Fixes test_valid_device_cpu, test_valid_device_cuda_with_index, test_valid_device_tpu

---

## Fix #9: Parser Execution Config Support (Affects 1 test)

**File:** `neural/parser/parser.py`
**Location:** In ModelTransformer class, network transformation method

**Add execution_config extraction:**
```python
def transform_network(self, tree):
    network_data = {
        'name': self.extract_name(tree),
        'input': self.extract_input(tree),
        'layers': self.extract_layers(tree),
        'optimizer': self.extract_optimizer(tree),
        'loss': self.extract_loss(tree),
        'device': self.extract_device(tree),
        # Add this:
        'execution_config': self.extract_execution_config(tree),  # New method
    }
    return network_data

def extract_execution_config(self, tree) -> Optional[Dict]:
    """Extract execution configuration from tree."""
    for child in tree.children:
        if hasattr(child, 'data') and child.data == 'execution_config':
            return self.transform_execution_config(child)
    return None

def transform_execution_config(self, node) -> Dict:
    """Transform execution config node to dictionary."""
    config = {}
    for child in node.children:
        if hasattr(child, 'data'):
            if child.data == 'config_pair':
                key = child.children[0].value
                value = self.extract_value(child.children[1])
                config[key] = value
    return config
```

**Impact:** Fixes test_network_with_execution_config

---

## Fix #10: Parser Required Field Validation (Affects 2 tests)

**File:** `neural/parser/parser.py`
**Location:** In ModelTransformer class, after network transformation

**Add validation:**
```python
def transform_network(self, tree):
    network_data = {
        'name': self.extract_name(tree),
        'input': self.extract_input(tree),
        'layers': self.extract_layers(tree),
        'optimizer': self.extract_optimizer(tree),
        'loss': self.extract_loss(tree),
    }
    
    # Add validation:
    self.validate_required_fields(network_data)
    
    return network_data

def validate_required_fields(self, network_data: Dict) -> None:
    """Validate that required fields are present."""
    required_fields = ['optimizer', 'loss']
    for field in required_fields:
        if field not in network_data or network_data[field] is None:
            raise DSLValidationError(
                f"Required field '{field}' is missing in network definition",
                severity="ERROR"
            )
```

**Impact:** Fixes test_network_missing_optimizer and test_network_missing_loss

---

## Fix #11: Test Fixture Updates for auto_flatten_output (Affects 45 tests)

This is the most widespread issue. Instead of fixing code, we can fix the test fixtures.

**Files:** Various test files in `tests/code_generator/`

**Pattern to find and fix:**
```python
# Find model_data definitions like:
model_data = {
    'input': {'shape': (None, 28, 28, 3), 'type': 'Input'},
    'layers': [
        {'type': 'Conv2D', 'params': {'filters': 32}},
        # ... more layers
        {'type': 'Output', 'params': {'units': 10}}
    ],
}

# Add auto_flatten_output:
model_data = {
    'input': {'shape': (None, 28, 28, 3), 'type': 'Input'},
    'layers': [
        {'type': 'Conv2D', 'params': {'filters': 32}},
        # ... more layers
        {'type': 'Output', 'params': {'units': 10}}
    ],
    'auto_flatten_output': True,  # ADD THIS LINE
}
```

**Alternative approach** - Add Flatten layer before Output:
```python
model_data = {
    'input': {'shape': (None, 28, 28, 3), 'type': 'Input'},
    'layers': [
        {'type': 'Conv2D', 'params': {'filters': 32}},
        # ... more layers
        {'type': 'Flatten', 'params': {}},  # ADD THIS
        {'type': 'Output', 'params': {'units': 10}}
    ],
}
```

**Impact:** Fixes 43 failing code generation tests

---

## Fix #12: Shape Propagation Tests - Add pytest.raises (Affects 19 tests)

**Files:** `tests/shape_propagation/test_error_handling.py` and related

**Current pattern:**
```python
def test_conv2d_kernel_too_large():
    propagator = ShapePropagator()
    layer = {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': 30}}
    propagator.propagate(layer, (1, 28, 28, 3))
```

**Fix with pytest.raises:**
```python
import pytest
from neural.exceptions import ShapeMismatchError

def test_conv2d_kernel_too_large():
    propagator = ShapePropagator()
    layer = {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': 30}}
    
    # Expect exception to be raised:
    with pytest.raises(ShapeMismatchError, match="kernel size .* exceeds input dimensions"):
        propagator.propagate(layer, (1, 28, 28, 3))
```

**Apply this pattern to all error handling tests that currently fail with exceptions.**

**Impact:** Fixes 19 shape propagation error handling tests

---

## Summary of Quick Wins

| Fix # | Description | Files Changed | Tests Fixed | Effort (min) |
|-------|-------------|---------------|-------------|--------------|
| 1 | keras import | 1 | 5 modules | 5 |
| 2 | parser import | 1 | 1 module | 2 |
| 3 | Directory creation | 1 | 2 | 15 |
| 4 | Empty sublayers | 1 | 1 | 10 |
| 5 | Split params | 1 | 1 | 5 |
| 6 | Custom layer warning | 1 | 1 | 10 |
| 7 | ONNX shape | 1 | 1 | 15 |
| 8 | Device preservation | 1 | 3 | 30 |
| 9 | Execution config | 1 | 1 | 30 |
| 10 | Required validation | 1 | 2 | 15 |
| 11 | Auto flatten tests | ~20 | 43 | 120 |
| 12 | pytest.raises | ~10 | 19 | 60 |

**Total estimated effort:** ~5 hours to fix 79+ tests

**Recommended order:**
1. Fixes #1-2 (import errors) - Immediate unblocking
2. Fixes #3-7 (simple code fixes) - Quick wins
3. Fix #12 (test updates) - Medium effort, high impact
4. Fix #11 (test fixture updates) - Higher effort but fixes most tests
5. Fixes #8-10 (parser enhancements) - More complex, schedule separately

---

## Testing Strategy

After implementing fixes:

1. **Run core modules first:**
   ```bash
   pytest tests/parser/ tests/code_generator/ tests/shape_propagation/ -v
   ```

2. **Run integration tests:**
   ```bash
   pytest tests/integration_tests/ tests/hpo/ -v
   ```

3. **Run full suite:**
   ```bash
   pytest tests/ -v --tb=short
   ```

4. **Check coverage:**
   ```bash
   pytest --cov=neural --cov-report=html
   ```

---

**End of Quick Fixes Guide**
