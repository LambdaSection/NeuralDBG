"""
Parameter validation utilities for Neural DSL parser.
Provides strict type checking and conversion for layer parameters.
"""

from typing import Union, Any, TypeVar, Optional, Type
from enum import Enum
import numpy as np

class ParamType(Enum):
    """Enumeration of parameter types supported in Neural DSL."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "array"
    DICT = "dict"

T = TypeVar('T')

class ValidationError(Exception):
    """Exception raised for parameter validation errors."""
    pass

def validate_numeric(
    value: Any,
    param_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    integer_only: bool = False
) -> Union[int, float]:
    """
    Validate and convert a value to a numeric type.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        integer_only: Whether only integer values are allowed
        
    Returns:
        Validated and converted numeric value
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(value, (dict, list)):
            raise ValidationError(f"{param_name} must be a number, got {type(value).__name__}")
        
        # Try to convert to float first
        if isinstance(value, str):
            # Remove any whitespace
            value = value.strip()
        
        num_val = float(value)
        
        # Check if we need an integer
        if integer_only:
            if not float(num_val).is_integer():
                raise ValidationError(f"{param_name} must be an integer, got {num_val}")
            num_val = int(num_val)
            
        # Check bounds
        if min_value is not None and num_val < min_value:
            raise ValidationError(f"{param_name} must be >= {min_value}, got {num_val}")
        if max_value is not None and num_val > max_value:
            raise ValidationError(f"{param_name} must be <= {max_value}, got {num_val}")
            
        return int(num_val) if integer_only else num_val
        
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Invalid {param_name}: {str(e)}")

def validate_units(
    value: Any,
    param_name: str = "units"
) -> int:
    """
    Validate and convert a units parameter value.
    Units must be positive integers.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        
    Returns:
        Validated units value as integer
        
    Raises:
        ValidationError: If validation fails
    """
    result = validate_numeric(
        value,
        param_name,
        min_value=1,
        integer_only=True
    )
    return int(result)  # Safe cast since we specified integer_only=True

def validate_shape(
    value: Any,
    param_name: str = "shape"
) -> tuple[int, ...]:
    """
    Validate and convert a shape parameter value.
    Shape dimensions must be positive integers.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        
    Returns:
        Validated shape as tuple of integers
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (list, tuple)):
        raise ValidationError(f"{param_name} must be a list or tuple, got {type(value).__name__}")
        
    try:
        dims = [int(validate_numeric(dim, f"{param_name} dimension", min_value=1, integer_only=True))
                for dim in value]
        return tuple(dims)
    except ValidationError as e:
        raise ValidationError(f"Invalid {param_name}: {str(e)}")

def validate_probability(
    value: Any,
    param_name: str = "probability"
) -> float:
    """
    Validate and convert a probability parameter value.
    Must be a float between 0 and 1.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error messages
        
    Returns:
        Validated probability as float
        
    Raises:
        ValidationError: If validation fails
    """
    return validate_numeric(
        value,
        param_name,
        min_value=0.0,
        max_value=1.0
    )