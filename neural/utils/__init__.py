"""
Utilities for Neural DSL.

This module provides common utility functions including deterministic seeding
for reproducible model training and evaluation, and centralized logging.

Functions
---------
set_seed
    Set seeds across all frameworks for reproducibility
set_global_seed
    Alternative seeding function with auto-generation
get_current_seed
    Retrieve current random seed value
get_logger
    Get a configured logger instance
setup_logging
    Configure global logging settings

Examples
--------
>>> from neural.utils import set_seed
>>> applied = set_seed(42, deterministic=True)
>>> print(applied)
{'python': True, 'numpy': True, 'torch': True, 'tensorflow': True}

>>> from neural.utils import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Processing started")
"""

from .seed import set_seed
from .seeding import set_global_seed, get_current_seed
from .logging import (
    get_logger,
    setup_logging,
    set_log_level,
    disable_logging,
    enable_logging,
    LogLevel,
    LogContext,
    log_function_call
)

__all__ = [
    'set_seed',
    'set_global_seed',
    'get_current_seed',
    'get_logger',
    'setup_logging',
    'set_log_level',
    'disable_logging',
    'enable_logging',
    'LogLevel',
    'LogContext',
    'log_function_call'
]
