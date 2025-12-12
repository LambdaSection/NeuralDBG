"""
Utilities for Neural DSL.

This module provides common utility functions including deterministic seeding
for reproducible model training and evaluation.

Functions
---------
set_seed
    Set seeds across all frameworks for reproducibility
set_global_seed
    Alternative seeding function with auto-generation
get_current_seed
    Retrieve current random seed value

Examples
--------
>>> from neural.utils import set_seed
>>> applied = set_seed(42, deterministic=True)
>>> print(applied)
{'python': True, 'numpy': True, 'torch': True, 'tensorflow': True}
"""

from .seed import set_seed
from .seeding import set_global_seed, get_current_seed

__all__ = ['set_seed', 'set_global_seed', 'get_current_seed']
