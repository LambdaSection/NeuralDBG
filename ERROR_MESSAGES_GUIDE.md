# Enhanced Error Messages Guide

This document describes the enhanced error messaging system in Neural DSL, which provides actionable suggestions and diagnostic information to help users quickly identify and fix issues.

## Overview

The error messaging system has been enhanced in two main areas:

1. **Parser Error Handling** (`neural/parser/error_handling.py`)
2. **Shape Propagation Diagnostics** (`neural/shape_propagation/shape_propagator.py`)

## Parser Error Enhancements

### Features

#### 1. Typo Detection and Correction
The system detects common typos in layer names, parameters, and network properties:

```python
# Input (with typo):
network MyModel {
  layers:
    Desnse(units=128)  # Typo: "Desnse"
}

# Error Message:
❌ Unexpected token 'Desnse' at line 3, column 5
💡 Suggestion: Did you mean 'Dense'?
🔧 Fix: Replace 'Desnse' with 'Dense'
```

#### 2. Context-Aware Issue Detection
Detects common DSL syntax issues:
- Missing colons after network properties
- Unclosed parentheses and braces
- Missing quotes around string values
- Incorrect parameter syntax

```python
# Missing colon:
network MyModel {
  layers
    Dense(128)
}

# Error:
⚠️ Common Issue: Add a colon ':' after 'layers' to define the network property
```

#### 3. Visual Error Indicators
Shows code context with visual arrows pointing to the error:

```
📄 Context:
    2  | network MyModel {
    3  |   input: (28, 28, 1)
    4  |   layers:
>>> 5  |     Desnse(units=128)
              ^~~~~~
    6  |     Output(10)
```

#### 4. Quick Reference Guide
Each error includes a quick reference for correct syntax:

```
📚 Quick Reference:
   - Layer syntax: LayerName(param1=value1, param2=value2)
   - Network properties: input:, layers:, loss:, optimizer:
   - Strings must be quoted: activation="relu"
```

### Supported Error Types

1. **Syntax Errors**: Unexpected tokens, characters, missing punctuation
2. **Typo Detection**: Common misspellings of 70+ layer names and parameters
3. **Structural Errors**: Missing colons, unclosed brackets, malformed tuples

## Shape Propagation Enhancements

### Features

#### 1. Detailed Parameter Validation
Enhanced validation for layer parameters with specific error messages:

```python
# Missing required parameter:
Conv2D(kernel_size=(3,3))  # Missing 'filters'

# Error Message:
======================================================================
SHAPE ERROR: Conv2D Layer
======================================================================

❌ Conv2D layer requires 'filters' parameter

🔧 Fix Suggestions:
   1. Add filters parameter: Conv2D(filters=32, ...)
   2. Common values: 32, 64, 128, 256 for different network depths
   3. Example: Conv2D(filters=32, kernel_size=(3,3), activation='relu')

💡 Tip: Use 'neural visualize' to see layer shapes throughout your network
======================================================================
```

#### 2. Shape Incompatibility Detection
Identifies when layer parameters don't match input dimensions:

```python
# Kernel size too large:
Conv2D(filters=32, kernel_size=(5,5))
# With input shape: (None, 3, 3, 1)

# Error Message:
❌ Conv2D kernel size (5, 5) exceeds input dimensions (3, 3)

📊 Input Shape: (None, 3, 3, 1)
   Expected: kernel_size <= (3, 3)
   Got: kernel_size = (5, 5)

🔧 Fix Suggestions:
   1. Reduce kernel_size to fit within (3, 3)
   2. Try kernel_size=(3, 3)
   3. Or increase input size before this layer
   4. Add padding if you need larger receptive field
```

#### 3. Dimensional Mismatch Guidance
Provides specific guidance for dimension mismatches:

```python
# Dense after Conv without Flatten:
Conv2D(filters=32, kernel_size=(3,3))
Dense(units=128)  # Error: expects 2D input

# Error Message:
❌ Dense layer expects 2D input (batch, features), got 4D: (None, 26, 26, 32)

📊 Input Shape: (None, 26, 26, 32)
   Expected: (batch_size, features)

🔧 Fix Suggestions:
   1. Add a Flatten() layer before Dense to convert multi-dimensional input to 2D
   2. Example sequence: Conv2D -> MaxPooling2D -> Flatten() -> Dense
   3. Your input shape (None, 26, 26, 32) needs flattening first
   4. Or use GlobalAveragePooling2D instead of Flatten for spatial data
```

### Error Categories

1. **missing_parameter**: Required parameter not provided
2. **invalid_parameter**: Parameter has invalid value (negative, zero, wrong type)
3. **shape_incompatibility**: Input shape doesn't match layer requirements
4. **malformed_layer**: Layer structure is incorrect
5. **invalid_input_shape**: Input shape has invalid dimensions

## Utility Functions

### Enhanced Error Formatting (`neural/shape_propagation/utils.py`)

#### `format_error_message(error_type, details)`
Creates formatted error messages with fix suggestions for common issues.

#### `suggest_layer_fix(layer_type, error_context)`
Provides layer-specific fix recommendations based on:
- Layer type
- Input shape
- Parameter values

#### `diagnose_shape_flow(shape_history)`
Analyzes the entire network's shape flow to identify:
- Dimension collapses
- Extreme size reductions (>99% reduction)
- Potential bottlenecks
- Memory usage issues

## Usage Examples

### Example 1: Typo in Layer Name
```neural
network Example {
  input: (28, 28, 1)
  layers:
    Conv2d(filters=32, kernel_size=(3,3))  # lowercase 'd'
}
```

**Error Message:**
```
❌ Unexpected token 'Conv2d'
💡 Suggestion: Did you mean 'Conv2D'?
🔧 Fix: Replace 'Conv2d' with 'Conv2D'
```

### Example 2: Missing Parameter
```neural
network Example {
  input: (28, 28, 1)
  layers:
    Dense(128)  # Missing 'units=' prefix
}
```

**Error Message:**
```
❌ Dense layer requires 'units' parameter

🔧 Fix Suggestions:
   1. Add units parameter: Dense(units=128)
   2. Common values: 64, 128, 256, 512 for hidden layers
   3. Example: Dense(units=128, activation='relu')
```

### Example 3: Shape Mismatch
```neural
network Example {
  input: (10, 10, 3)
  layers:
    MaxPooling2D(pool_size=(12, 12))  # Pool size > input
}
```

**Error Message:**
```
❌ MaxPooling2D pool_size (12, 12) exceeds input dimensions (10, 10)

📊 Input Shape: (None, 10, 10, 3)
   Expected: pool_size <= (10, 10)
   Got: pool_size = (12, 12)

🔧 Fix Suggestions:
   1. Reduce pool_size to fit within (10, 10)
   2. Try pool_size=(2, 2)
   3. Or remove this pooling layer if input is already small
   4. Current spatial dimensions: (10, 10)
```

## Common Fix Patterns

### Pattern 1: Dense After Conv Layers
**Problem:** Dense layer receives multi-dimensional input

**Solution:**
```neural
# Wrong:
Conv2D(filters=32, kernel_size=(3,3))
Dense(units=128)

# Correct:
Conv2D(filters=32, kernel_size=(3,3))
Flatten()
Dense(units=128)
```

### Pattern 2: Kernel Size Too Large
**Problem:** Kernel size exceeds input dimensions

**Solution:**
```neural
# Wrong (with 10x10 input):
Conv2D(filters=32, kernel_size=(12,12))

# Correct:
Conv2D(filters=32, kernel_size=(3,3))
# Or add padding:
Conv2D(filters=32, kernel_size=(12,12), padding='same')
```

### Pattern 3: Missing Required Parameters
**Problem:** Layer missing required parameters

**Solution:**
```neural
# Wrong:
Conv2D(kernel_size=(3,3))  # Missing filters

# Correct:
Conv2D(filters=32, kernel_size=(3,3))
```

## Best Practices

1. **Use Descriptive Names**: Follow exact case for layer names (Conv2D, not conv2d)
2. **Always Quote Strings**: activation="relu", not activation=relu
3. **Check Dimensions**: Use `neural visualize` to see shape flow
4. **Start Simple**: Build networks incrementally to catch errors early
5. **Read Error Messages**: Enhanced messages provide specific solutions

## Diagnostic Tools

### Shape Flow Diagnosis
```python
propagator = ShapePropagator()
# ... propagate through layers ...
diagnostics = propagator.diagnose_shape_issues()

# Returns:
{
    'warnings': [...],      # Non-critical issues
    'errors': [...],        # Critical issues
    'suggestions': [...],   # Overall recommendations
    'shape_flow': [...]     # Detailed shape progression
}
```

### Layer-Specific Suggestions
```python
suggestions = propagator.get_layer_fix_suggestions(
    layer_type='Conv2D',
    input_shape=(None, 28, 28, 1),
    params={'filters': 32, 'kernel_size': (50, 50)}
)
# Returns list of actionable fix suggestions
```

## Implementation Details

### Parser Error Handler
- **Location**: `neural/parser/error_handling.py`
- **Key Class**: `ErrorHandler`
- **Main Methods**:
  - `handle_unexpected_token()`: Processes Lark parsing errors
  - `suggest_correction()`: Finds similar valid tokens using fuzzy matching
  - `detect_common_issues()`: Identifies structural problems

### Shape Propagation Error Handler
- **Location**: `neural/shape_propagation/shape_propagator.py`
- **Key Class**: `ShapeMismatchError`
- **Main Methods**:
  - `_validate_layer_params()`: Enhanced validation with detailed errors
  - `diagnose_shape_issues()`: Network-wide shape analysis
  - `get_layer_fix_suggestions()`: Layer-specific recommendations

### Utility Functions
- **Location**: `neural/shape_propagation/utils.py`
- **Key Functions**:
  - `format_error_message()`: Formats errors with suggestions
  - `suggest_layer_fix()`: Provides fix recommendations
  - `diagnose_shape_flow()`: Analyzes shape progression
