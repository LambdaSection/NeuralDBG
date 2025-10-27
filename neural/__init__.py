"""
Neural DSL - A domain-specific language for neural networks.

This package provides a declarative syntax for defining, training, debugging,
and deploying neural networks with cross-framework support.
"""

__version__ = "0.3.0.dev0"
__author__ = "Lemniscate-SHA-256/SENOUVO Jacques-Charles Gad"
__email__ = "Lemniscate_zero@proton.me"

# Core modules
from . import cli
from . import parser
from . import shape_propagation
from . import code_generation
from . import visualization
from . import dashboard
from . import hpo
from . import cloud

# Utility modules
from . import utils

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "cli",
    "parser", 
    "shape_propagation",
    "code_generation",
    "visualization",
    "dashboard",
    "hpo",
    "cloud",
    "utils"
]
