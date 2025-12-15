# Error Suggestions Quick Reference

## API Reference

### ErrorSuggestion Class

All methods are static and return `Optional[str]`:
- Return suggestion string when input has issues
- Return `None` when input is correct

#### Parameter Validation

```python
ErrorSuggestion.suggest_parameter_fix(param_name: str, layer_type: str) -> Optional[str]
```

**Examples:**
```python
suggest_parameter_fix("unit", "Dense")    # → "Did you mean 'units'...?"
suggest_parameter_fix("units", "Dense")   # → None (correct)
suggest_parameter_fix("filter", "Conv2D") # → "Did you mean 'filters'...?"
```

**Common typos detected:**
- `unit` → `units`
- `filter` → `filters`
- `kernal_size` → `kernel_size`
- `activaton` → `activation`

#### Layer Name Validation

```python
ErrorSuggestion.suggest_layer_fix(layer_name: str) -> Optional[str]
```

**Examples:**
```python
suggest_layer_fix("dense")         # → "Layer names are case-sensitive..."
suggest_layer_fix("Dense")         # → None (correct)
suggest_layer_fix("MaxPool2D")     # → "Did you mean 'MaxPooling2D'?"
suggest_layer_fix("MaxPooling2D")  # → None (correct)
```

**Common issues detected:**
- Case sensitivity: `dense` → `Dense`, `lstm` → `LSTM`
- Aliases: `MaxPool2D` → `MaxPooling2D`, `BatchNorm` → `BatchNormalization`

#### Shape Validation

```python
ErrorSuggestion.suggest_shape_fix(input_shape: tuple, layer_type: str) -> Optional[str]
```

**Examples:**
```python
suggest_shape_fix((None, -28, 28), "Dense")      # → "Shape dimensions cannot be negative..."
suggest_shape_fix((None, 28, 28, 1), "Conv2D")   # → None (valid)
suggest_shape_fix((None, 28, 28, 1), "Dense")    # → "Dense layers expect 2D input..."
```

**Issues detected:**
- Negative dimensions
- Empty shapes
- Wrong dimensionality for layer type

#### Parameter Value Validation

```python
ErrorSuggestion.suggest_parameter_value_fix(param: str, value: Any, layer_type: str) -> Optional[str]
```

**Examples:**
```python
suggest_parameter_value_fix("units", -10, "Dense")    # → "Units must be positive..."
suggest_parameter_value_fix("units", 128, "Dense")    # → None (valid)
suggest_parameter_value_fix("rate", 1.5, "Dropout")   # → "Dropout rate must be between 0 and 1..."
```

**Constraints checked:**
- `units`, `filters` > 0
- `rate` (Dropout) ∈ [0, 1)
- `kernel_size`, `pool_size` > 0
- `learning_rate` > 0

#### Activation Function Validation

```python
ErrorSuggestion.suggest_activation_fix(activation: str) -> Optional[str]
```

**Examples:**
```python
suggest_activation_fix("Relu")      # → "Use 'relu' (lowercase)"
suggest_activation_fix("relu")      # → None (correct)
suggest_activation_fix("leakyrelu") # → "Did you mean 'leaky_relu'?"
```

#### Optimizer Name Validation

```python
ErrorSuggestion.suggest_optimizer_fix(optimizer: str) -> Optional[str]
```

**Examples:**
```python
suggest_optimizer_fix("adam")    # → "Did you mean 'Adam'...?"
suggest_optimizer_fix("Adam")    # → None (correct)
suggest_optimizer_fix("sgd")     # → "Did you mean 'SGD'...?"
```

#### Loss Function Validation

```python
ErrorSuggestion.suggest_loss_fix(loss: str) -> Optional[str]
```

**Examples:**
```python
suggest_loss_fix("crossentropy")           # → "Did you mean 'categorical_crossentropy'?"
suggest_loss_fix("categorical_crossentropy") # → None (correct)
```

#### Backend Validation

```python
ErrorSuggestion.suggest_backend_fix(backend: str) -> str
```

**Examples:**
```python
suggest_backend_fix("tf")      # → "Use 'tensorflow' instead..."
suggest_backend_fix("torch")   # → "Use 'pytorch' instead..."
```

#### Dependency Installation

```python
ErrorSuggestion.suggest_dependency_install(dependency: str) -> str
```

**Examples:**
```python
suggest_dependency_install("torch")  # → "Missing dependency: torch. Install with: pip install torch torchvision"
suggest_dependency_install("optuna") # → "...Install with: pip install 'neural-dsl[hpo]'"
```

### ErrorFormatter Class

#### Parser Errors

```python
ErrorFormatter.format_parser_error(
    message: str,
    line: Optional[int] = None,
    column: Optional[int] = None,
    code_snippet: Optional[str] = None,
    suggestion: Optional[str] = None
) -> str
```

#### Shape Errors

```python
ErrorFormatter.format_shape_error(
    message: str,
    input_shape: Optional[tuple] = None,
    expected_shape: Optional[tuple] = None,
    layer_type: Optional[str] = None,
    suggestion: Optional[str] = None
) -> str
```

#### Parameter Errors

```python
ErrorFormatter.format_parameter_error(
    parameter: str,
    value: Any,
    layer_type: str,
    reason: Optional[str] = None,
    expected: Optional[str] = None,
    suggestion: Optional[str] = None
) -> str
```

Auto-generates suggestions if not provided.

#### Dependency Errors

```python
ErrorFormatter.format_dependency_error(
    dependency: str,
    feature: Optional[str] = None,
    import_error: Optional[str] = None
) -> str
```

Auto-generates installation instructions.

## Testing Guidelines

### Test Structure

```python
def test_suggestion_for_error():
    """Test that errors receive suggestions."""
    result = ErrorSuggestion.suggest_XXX(incorrect_input)
    assert result is not None
    assert "expected_text" in result

def test_no_suggestion_for_correct():
    """Test that correct inputs receive no suggestions."""
    result = ErrorSuggestion.suggest_XXX(correct_input)
    assert result is None
```

### Common Assertions

```python
# Error case
assert suggestion is not None
assert "expected_correction" in suggestion
assert "💡" in formatted_error

# Correct case
assert suggestion is None
```

## Design Principles

1. **Fail Silent for Correct Inputs**: No suggestion = no problem
2. **Actionable Suggestions**: Tell users exactly what to fix
3. **Context-Aware**: Use layer type, parameter name, and value to provide specific guidance
4. **Auto-Generation**: Formatters auto-generate suggestions when not explicitly provided

## See Also

- [Full Documentation](troubleshooting.md#error-suggestions-system)
- [Implementation Summary](../ERROR_SUGGESTIONS_FIX_SUMMARY.md)
- [Test Suite](../tests/test_error_suggestions.py)
