# Neural DSL Logging Guide

This guide explains how to use the centralized logging system in Neural DSL.

## Quick Start

```python
from neural.utils import get_logger

# Create a logger for your module
logger = get_logger(__name__)

# Use it throughout your module
logger.debug("Detailed debug information")
logger.info("General informational message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error message")
```

## Basic Usage

### Creating a Logger

Always create a module-level logger using `__name__`:

```python
from neural.utils.logging import get_logger

logger = get_logger(__name__)
```

This ensures the logger name matches your module's import path (e.g., `neural.parser.parser`).

### Logging Messages

```python
# Debug - detailed diagnostic information
logger.debug(f"Processing layer: {layer_name}")

# Info - general informational messages
logger.info("Model compilation completed successfully")

# Warning - recoverable issues or deprecation warnings
logger.warning("Using deprecated parameter 'old_param'")

# Error - errors that might still allow the program to continue
logger.error(f"Failed to load model: {e}", exc_info=True)

# Critical - serious errors that might cause termination
logger.critical("Out of memory - cannot continue")
```

### Exception Logging

Include exception information automatically:

```python
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", exc_info=True)
    raise
```

## Configuration

### Global Setup

Configure logging for the entire application:

```python
from neural.utils import setup_logging, LogLevel

# Basic setup
setup_logging(level=LogLevel.INFO)

# With file output
setup_logging(
    level=LogLevel.DEBUG,
    log_file='neural.log'
)

# Custom format
setup_logging(
    level=LogLevel.INFO,
    format_string='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
```

### Per-Module Control

Set different log levels for different modules:

```python
from neural.utils import set_log_level, LogLevel

# Make parser very verbose
set_log_level('neural.parser', LogLevel.DEBUG)

# Silence code generation
set_log_level('neural.code_generation', LogLevel.ERROR)
```

### Temporary Log Level

Use a context manager for temporary log level changes:

```python
from neural.utils import LogContext, LogLevel, get_logger

logger = get_logger(__name__)

# Temporarily enable debug logging
with LogContext(logger, LogLevel.DEBUG):
    logger.debug("This will be logged")
    complex_operation()

# Back to normal logging level
logger.debug("This might not be logged")
```

## Advanced Features

### Function Call Logging

Automatically log function calls:

```python
from neural.utils import log_function_call, get_logger

logger = get_logger(__name__)

@log_function_call(logger)
def process_data(x, y):
    return x + y

# Logs: "Calling process_data with args=(1, 2), kwargs={}"
# Logs: "process_data returned 3"
result = process_data(1, 2)
```

### Disabling/Enabling Logging

```python
from neural.utils import disable_logging, enable_logging, LogLevel

# Silence a specific logger
disable_logging('neural.parser')

# Silence all logging
disable_logging()

# Re-enable
enable_logging('neural.parser', LogLevel.INFO)
enable_logging()  # Re-enable all
```

## Best Practices

### 1. Use Appropriate Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Something unexpected happened, but the program can continue
- **ERROR**: A serious problem occurred, but the program can still function
- **CRITICAL**: A very serious error that might cause termination

### 2. Include Context

```python
# Bad
logger.error("Failed")

# Good
logger.error(f"Failed to compile model '{model_name}': {error_message}")
```

### 3. Use Lazy Formatting

```python
# Bad - string is formatted even if not logged
logger.debug("Result: " + expensive_computation())

# Good - only formatted if actually logged
logger.debug("Result: %s", expensive_computation())

# Also good - f-strings are fine since Python 3.6+
logger.debug(f"Result: {result}")
```

### 4. Log Exceptions Properly

```python
try:
    dangerous_operation()
except ValueError as e:
    # Include exception info automatically
    logger.error("Operation failed", exc_info=True)
    # Or with custom message
    logger.exception("Operation failed")  # Automatically includes exc_info
    raise
```

### 5. Don't Log Sensitive Information

```python
# Bad
logger.info(f"User password: {password}")

# Good
logger.info(f"User authenticated: {username}")
```

## Migration from print()

### Before
```python
print(f"Processing {item}")
print(f"Warning: deprecated feature")
print(f"Error: {e}")
```

### After
```python
logger.info(f"Processing {item}")
logger.warning("Deprecated feature")
logger.error(f"Error: {e}", exc_info=True)
```

## Examples

### Module Template

```python
"""My module docstring."""
from __future__ import annotations

from typing import Optional

from neural.utils import get_logger

logger = get_logger(__name__)


def my_function(param: str) -> Optional[str]:
    """Process parameter and return result."""
    logger.debug(f"Processing parameter: {param}")
    
    try:
        result = complex_operation(param)
        logger.info("Processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Processing failed for {param}", exc_info=True)
        return None
```

### CLI Tool

```python
import click
from neural.utils import setup_logging, LogLevel, get_logger

logger = get_logger(__name__)


@click.command()
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def main(verbose: bool):
    """My CLI tool."""
    # Configure logging based on verbosity
    setup_logging(level=LogLevel.DEBUG if verbose else LogLevel.INFO)
    
    logger.info("Starting processing")
    # ... rest of your code
```

### Testing

```python
import logging
from neural.utils import get_logger

logger = get_logger(__name__)


def test_my_function(caplog):
    """Test with log capture."""
    with caplog.at_level(logging.DEBUG):
        my_function()
        assert "Processing completed" in caplog.text
```

## Colored Output

The logging system automatically provides colored output when running in a terminal:
- DEBUG: Cyan
- INFO: Green
- WARNING: Yellow
- ERROR: Red
- CRITICAL: Magenta

Colors are automatically disabled when output is redirected to a file.

## Performance Considerations

1. **Conditional Logging**: Debug statements are only evaluated if debug logging is enabled
2. **Lazy Formatting**: Use `%s` formatting or f-strings to avoid unnecessary string operations
3. **Log Level Checks**: For expensive operations, check log level first:

```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Expensive computation: {expensive_function()}")
```

## Troubleshooting

### Logs Not Appearing

1. Check log level: `set_log_level('neural', LogLevel.DEBUG)`
2. Ensure logger is created: `logger = get_logger(__name__)`
3. Check if logging was disabled: `enable_logging()`

### Too Much Output

1. Increase log level: `setup_logging(level=LogLevel.WARNING)`
2. Silence specific modules: `set_log_level('neural.verbose_module', LogLevel.ERROR)`

### Log File Issues

1. Ensure directory exists: `mkdir -p logs`
2. Check file permissions
3. Use absolute path: `setup_logging(log_file='/absolute/path/to/file.log')`

## Further Reading

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
