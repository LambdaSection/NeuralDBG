# Docstring Style Guide for Neural DSL

This guide provides templates and examples for writing consistent, comprehensive docstrings in Neural DSL.

## General Guidelines

1. **Always document** public functions, classes, and methods
2. **Use NumPy style** for consistency across the codebase
3. **Include type hints** in function signatures (not in docstrings)
4. **Provide examples** when helpful
5. **Keep descriptions concise** but complete
6. **Use imperative mood** ("Return the value", not "Returns the value")

## Function Template

```python
def function_name(param1: int, param2: str = "default") -> bool:
    """
    Brief one-line description (imperative mood).
    
    More detailed description if needed. Explain what the function does,
    not how it does it. Focus on the user's perspective.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str, optional
        Description of param2, by default "default"
        
    Returns
    -------
    bool
        Description of return value
        
    Raises
    ------
    ValueError
        When param1 is negative
    TypeError
        When param2 is not a string
        
    Examples
    --------
    >>> function_name(42, "test")
    True
    >>> function_name(0)
    False
    
    Notes
    -----
    Additional information, algorithm details, or references.
    
    See Also
    --------
    related_function : Related functionality
    """
    return True
```

## Class Template

```python
class ClassName:
    """
    Brief one-line description of the class.
    
    More detailed description of what the class represents and its purpose.
    Explain the main responsibilities and use cases.
    
    Parameters
    ----------
    param1 : int
        Description of initialization parameter
    param2 : str, optional
        Description of optional parameter, by default "default"
        
    Attributes
    ----------
    attr1 : int
        Description of public attribute
    attr2 : list
        Description of another attribute
        
    Examples
    --------
    >>> obj = ClassName(42, "test")
    >>> obj.method()
    result
    
    Notes
    -----
    Important implementation details or usage notes.
    """
    
    def __init__(self, param1: int, param2: str = "default") -> None:
        """
        Initialize the class.
        
        Parameters
        ----------
        param1 : int
            Description
        param2 : str, optional
            Description, by default "default"
        """
        self.attr1 = param1
        self.attr2 = []
    
    def method(self) -> str:
        """
        Brief description of method.
        
        Returns
        -------
        str
            Description of return value
        """
        return "result"
```

## Module Template

```python
"""
Module name and brief description.

More detailed description of the module's purpose, functionality,
and main components.

Features
--------
- Feature 1
- Feature 2
- Feature 3

Classes
-------
ClassName
    Brief description of class

Functions
---------
function_name
    Brief description of function

Examples
--------
>>> from neural.module import ClassName
>>> obj = ClassName()
>>> obj.method()
result

Notes
-----
Important notes about the module.
"""
```

## Real Examples from Neural DSL

### Simple Function

```python
def to_number(x: str) -> Union[int, float]:
    """
    Convert a string to int or float.
    
    Parameters
    ----------
    x : str
        String representation of a number
        
    Returns
    -------
    Union[int, float]
        Integer if possible, otherwise float
        
    Examples
    --------
    >>> to_number("42")
    42
    >>> to_number("3.14")
    3.14
    """
    try:
        return int(x)
    except ValueError:
        return float(x)
```

### Complex Function

```python
def generate_code(
    model_data: Dict[str, Any],
    backend: str,
    best_params: Optional[Dict[str, Any]] = None,
    auto_flatten_output: bool = False
) -> str:
    """
    Generate executable code from parsed DSL model data.

    Parameters
    ----------
    model_data : Dict[str, Any]
        Parsed model dictionary containing 'layers', 'input', 'optimizer', etc.
    backend : str
        Target framework: 'tensorflow', 'pytorch', or 'onnx'
    best_params : Optional[Dict[str, Any]], optional
        Hyperparameters from HPO to inject into code, by default None
    auto_flatten_output : bool, optional
        Automatically insert flatten before dense layers if needed, by default False
        
    Returns
    -------
    str
        Generated Python code as string
        
    Raises
    ------
    ValueError
        If model_data format is invalid or backend is unsupported
        
    Examples
    --------
    >>> from neural.parser.parser import create_parser, ModelTransformer
    >>> parser = create_parser('network')
    >>> tree = parser.parse("Network Test { Input: shape=(1,28,28) Dense: units=10 }")
    >>> transformer = ModelTransformer()
    >>> model_data = transformer.transform(tree)
    >>> code = generate_code(model_data, 'tensorflow')
    >>> print(code)  # doctest: +SKIP
    import tensorflow as tf
    ...
    
    Notes
    -----
    The generated code includes:
    - Framework imports
    - Model architecture definition
    - Optimizer configuration
    - Experiment tracking integration
    - Training configuration (if provided)
    """
    # Implementation
```

### Class with Methods

```python
class ShapePropagator:
    """
    Propagate tensor shapes through neural network layers.
    
    This class performs shape inference, validates layer configurations,
    detects potential issues, and suggests optimizations for neural network
    architectures across different frameworks.
    
    Parameters
    ----------
    debug : bool, optional
        Enable debug output, by default False
        
    Attributes
    ----------
    shape_history : list
        History of shape transformations through layers
    layer_connections : list
        Connections between layers for visualization
    issues : list
        Detected architecture issues
    optimizations : list
        Suggested architecture optimizations
        
    Examples
    --------
    >>> from neural.shape_propagation import ShapePropagator
    >>> propagator = ShapePropagator(debug=False)
    >>> input_shape = (None, 1, 28, 28)
    >>> layer = {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3}}
    >>> output_shape = propagator.propagate(input_shape, layer)
    >>> print(output_shape)
    (None, 32, 26, 26)
    """
    
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.shape_history = []
        # ... more initialization
    
    def propagate(
        self,
        input_shape: Tuple[Optional[int], ...],
        layer: Dict[str, Any],
        framework: str = 'tensorflow'
    ) -> Tuple[Optional[int], ...]:
        """
        Propagate tensor shape through a single layer.
        
        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            Input tensor shape including batch dimension (None for dynamic)
        layer : Dict[str, Any]
            Layer configuration with 'type' and 'params' keys
        framework : str, optional
            Target framework ('tensorflow', 'pytorch', 'jax'), by default 'tensorflow'
            
        Returns
        -------
        Tuple[Optional[int], ...]
            Output tensor shape after layer transformation
            
        Raises
        ------
        KeyError
            If layer is missing 'type' field
        ValueError
            If input shape is invalid or layer parameters are incorrect
            
        Examples
        --------
        >>> propagator = ShapePropagator()
        >>> input_shape = (None, 32, 28, 28)
        >>> layer = {"type": "MaxPooling2D", "params": {"pool_size": 2}}
        >>> output_shape = propagator.propagate(input_shape, layer)
        >>> print(output_shape)
        (None, 32, 14, 14)
        """
        # Implementation
```

## Section Descriptions

### Parameters
- List all parameters with their types
- Mark optional parameters with "optional" and their default
- Provide clear, concise descriptions
- Mention units, ranges, or constraints if relevant

### Returns
- Specify the return type
- Describe what the returned value represents
- For complex returns, explain the structure

### Raises
- List all exceptions that can be raised
- Explain when and why each is raised
- Help users anticipate and handle errors

### Examples
- Provide realistic, runnable examples
- Show common use cases
- Use doctest format when possible
- Mark long outputs with `# doctest: +SKIP`

### Notes
- Additional context or implementation details
- Performance considerations
- Algorithm descriptions
- References to papers or documentation

### See Also
- Link to related functions/classes
- Brief description of the relationship

## Common Patterns

### Type Aliases

```python
from typing import Union, Optional, Dict, Any

# Use Union for multiple types
def func(x: Union[int, float]) -> bool:
    ...

# Use Optional for nullable values
def func(x: Optional[str] = None) -> int:
    ...

# Use Dict/List for collections
def func(data: Dict[str, Any]) -> List[int]:
    ...
```

### Deprecation Warning

```python
def old_function():
    """
    Old function description.
    
    .. deprecated:: 0.3.0
        Use :func:`new_function` instead.
    """
```

### Version Added

```python
def new_function():
    """
    New function description.
    
    .. versionadded:: 0.3.0
    """
```

## Tools for Checking Docstrings

```bash
# Check docstring coverage
pip install interrogate
interrogate neural/

# Validate docstring format
pip install pydocstyle
pydocstyle neural/

# Build docs to check for errors
cd docs
make html
```

## Quick Checklist

- [ ] One-line summary using imperative mood
- [ ] Detailed description if needed
- [ ] All parameters documented with types
- [ ] Return value documented with type
- [ ] Exceptions documented
- [ ] Examples provided (if helpful)
- [ ] Type hints in function signature
- [ ] Consistent with NumPy style
- [ ] Clear and concise language
- [ ] No spelling or grammar errors
