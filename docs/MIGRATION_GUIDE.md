# Migration Guide - Technical Debt Improvements

This guide helps developers migrate their code to use the new improved patterns.

## Overview

The Neural DSL codebase has been enhanced with:
1. Centralized logging system
2. Improved type hints
3. Better error messages
4. Consistent patterns

This guide shows how to migrate existing code to use these improvements.

## 1. Migrating from print() to Logging

### Pattern 1: Debug Information

**Before:**
```python
def process_model(model_data):
    print(f"Processing model with {len(model_data['layers'])} layers")
    # ... processing code
    print(f"Model processing complete")
```

**After:**
```python
from neural.utils import get_logger

logger = get_logger(__name__)

def process_model(model_data):
    logger.debug(f"Processing model with {len(model_data['layers'])} layers")
    # ... processing code
    logger.info("Model processing complete")
```

### Pattern 2: Error Messages

**Before:**
```python
try:
    risky_operation()
except Exception as e:
    print(f"Error: {e}")
```

**After:**
```python
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
```

### Pattern 3: Warning Messages

**Before:**
```python
if deprecated_param:
    print("Warning: 'old_param' is deprecated, use 'new_param' instead")
```

**After:**
```python
if deprecated_param:
    logger.warning("'old_param' is deprecated, use 'new_param' instead")
```

### Pattern 4: Informational Messages

**Before:**
```python
print("Starting training...")
print(f"Epoch {epoch}/{total_epochs}")
```

**After:**
```python
logger.info("Starting training...")
logger.info(f"Epoch {epoch}/{total_epochs}")
```

## 2. Adding Type Hints

### Pattern 1: Function Parameters and Returns

**Before:**
```python
def process_data(data, backend='tensorflow'):
    # ... code
    return result
```

**After:**
```python
from __future__ import annotations
from typing import Any, Dict, Optional

def process_data(data: Dict[str, Any], backend: str = 'tensorflow') -> Optional[Dict[str, Any]]:
    # ... code
    return result
```

### Pattern 2: Class Attributes

**Before:**
```python
class MyClass:
    def __init__(self, name):
        self.name = name
        self.data = None
        self.count = 0
```

**After:**
```python
from __future__ import annotations
from typing import Optional, Any

class MyClass:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.data: Optional[Any] = None
        self.count: int = 0
```

### Pattern 3: Complex Types

**Before:**
```python
def merge_configs(configs):
    result = {}
    for config in configs:
        result.update(config)
    return result
```

**After:**
```python
from __future__ import annotations
from typing import Dict, List, Any

def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for config in configs:
        result.update(config)
    return result
```

## 3. Module Structure Updates

### Pattern: Standard Module Template

**Before:**
```python
import json
import os
from pathlib import Path

# ... rest of code
```

**After:**
```python
"""
Module description.

This module provides...
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from neural.utils import get_logger

logger = get_logger(__name__)

# ... rest of code
```

## 4. Error Handling Improvements

### Pattern 1: Better Error Messages

**Before:**
```python
if not valid:
    raise ValueError("Invalid input")
```

**After:**
```python
from neural.exceptions import InvalidParameterError

if not valid:
    raise InvalidParameterError(
        parameter='input_data',
        value=input_data,
        expected='non-empty dictionary with "layers" key'
    )
```

### Pattern 2: Logging Before Raising

**Before:**
```python
if port < 1024:
    raise ValueError(f"Invalid port: {port}")
```

**After:**
```python
if port < 1024:
    logger.error(f"Invalid port number: {port} (minimum: 1024)")
    raise ValueError(f"Port must be >= 1024, got {port}")
```

## 5. Docstring Improvements

### Pattern: Comprehensive Docstrings

**Before:**
```python
def calculate_cost(hours, instance_type):
    """Calculate training cost."""
    # ... code
```

**After:**
```python
def calculate_cost(hours: float, instance_type: str) -> float:
    """
    Calculate training cost for cloud instances.
    
    Args:
        hours: Number of training hours
        instance_type: Cloud instance type (e.g., 'p3.2xlarge')
    
    Returns:
        Estimated cost in USD
    
    Raises:
        ValueError: If hours is negative or instance_type is unknown
    
    Examples:
        >>> cost = calculate_cost(10.5, 'p3.2xlarge')
        >>> print(f"${cost:.2f}")
        $31.50
    """
    # ... code
```

## 6. Configuration Patterns

### Pattern: Using LogLevel Enum

**Before:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**After:**
```python
from neural.utils import setup_logging, LogLevel

setup_logging(level=LogLevel.DEBUG)
```

## 7. Testing Patterns

### Pattern: Testing with Logging

**Before:**
```python
def test_my_function():
    result = my_function()
    assert result is not None
```

**After:**
```python
import logging
from neural.utils import get_logger

def test_my_function(caplog):
    with caplog.at_level(logging.INFO):
        result = my_function()
        assert result is not None
        assert "Processing completed" in caplog.text
```

## 8. Batch Migration Script

Here's a script to help with bulk migration:

```python
#!/usr/bin/env python
"""
Script to help migrate print() statements to logging.

Usage:
    python migrate_logging.py <file_or_directory>
"""
import re
import sys
from pathlib import Path

def migrate_file(filepath: Path) -> None:
    """Migrate a single file from print() to logging."""
    content = filepath.read_text()
    
    # Check if file already has logging
    if 'get_logger' in content or 'logger = logging.getLogger' in content:
        print(f"⏭️  {filepath} - Already using logging")
        return
    
    # Check if file has print statements
    if 'print(' not in content:
        print(f"⏭️  {filepath} - No print statements")
        return
    
    # Add import if not present
    if 'from neural.utils import get_logger' not in content:
        # Find first import or after module docstring
        lines = content.split('\n')
        insert_pos = 0
        
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i
                break
        
        lines.insert(insert_pos, 'from neural.utils import get_logger\n')
        lines.insert(insert_pos + 1, f'logger = get_logger(__name__)\n')
        content = '\n'.join(lines)
    
    # Replace common patterns
    content = re.sub(
        r'print\(f?"DEBUG:([^"]+)"\)',
        r'logger.debug(\1)',
        content
    )
    content = re.sub(
        r'print\(f?"Error:([^"]+)"\)',
        r'logger.error(\1)',
        content
    )
    content = re.sub(
        r'print\(f?"Warning:([^"]+)"\)',
        r'logger.warning(\1)',
        content
    )
    
    filepath.write_text(content)
    print(f"✅ {filepath} - Migrated")

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        migrate_file(path)
    elif path.is_dir():
        for pyfile in path.rglob('*.py'):
            migrate_file(pyfile)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

## 9. Checklist for New Code

When writing new code, ensure:

- [ ] Module has docstring
- [ ] `from __future__ import annotations` is imported
- [ ] Type hints are used for function parameters and returns
- [ ] `get_logger(__name__)` is used instead of print()
- [ ] Appropriate log levels are used (debug, info, warning, error)
- [ ] Exceptions include context and are logged before raising
- [ ] Complex types use proper type hints (Dict, List, Optional, etc.)
- [ ] Docstrings include Args, Returns, Raises, and Examples sections
- [ ] Error messages are descriptive and include relevant context

## 10. Common Pitfalls

### Pitfall 1: Mixing print() and logging

**Don't:**
```python
print("Starting...")
logger.info("Processing")
print("Done")
```

**Do:**
```python
logger.info("Starting...")
logger.info("Processing")
logger.info("Done")
```

### Pitfall 2: Wrong log level

**Don't:**
```python
logger.error("Processing item 5 of 100")  # Not an error!
```

**Do:**
```python
logger.debug("Processing item 5 of 100")  # Or info if important
```

### Pitfall 3: Missing exception info

**Don't:**
```python
except Exception as e:
    logger.error(str(e))  # Loses stack trace
```

**Do:**
```python
except Exception as e:
    logger.error("Operation failed", exc_info=True)  # Includes stack trace
```

### Pitfall 4: Overly verbose type hints

**Don't:**
```python
from typing import Dict, List, Tuple, Union, Optional, Any
def func() -> Union[Dict[str, List[Tuple[int, str]]], None]:
    pass
```

**Do:**
```python
from typing import Dict, List, Tuple, Optional
def func() -> Optional[Dict[str, List[Tuple[int, str]]]]:
    pass
```

## 11. Resources

- [Logging Guide](./LOGGING_GUIDE.md) - Detailed logging documentation
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [PEP 484](https://www.python.org/dev/peps/pep-0484/) - Type Hints
- [PEP 585](https://www.python.org/dev/peps/pep-0585/) - Type Hinting Generics

## 12. Getting Help

If you have questions about migrating your code:

1. Check the [Logging Guide](./LOGGING_GUIDE.md)
2. Look at examples in recently updated modules:
   - `neural/dashboard/dashboard.py`
   - `neural/execution_optimization/execution.py`
   - `neural/profiling/profiler_manager.py`
3. Review exception patterns in `neural/exceptions.py`

## Summary

The key changes to adopt:

1. **Replace `print()` → `logger.debug/info/warning/error/critical()`**
2. **Add type hints to functions and classes**
3. **Use descriptive error messages with context**
4. **Follow the module template structure**
5. **Write comprehensive docstrings**

These changes make the codebase more maintainable, debuggable, and professional.
