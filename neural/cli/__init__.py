"""
Neural CLI module.
This module provides the command-line interface for Neural.
"""

from .cli import cli, visualize

# Import additional modules for testing
try:
    from ..parser.parser import create_parser, ModelTransformer
    from ..shape_propagation.shape_propagator import ShapePropagator
    _parser_available = True
except ImportError:
    _parser_available = False

__all__ = ['cli', 'visualize']

# Export parser functions if available
if _parser_available:
    __all__.extend(['create_parser', 'ModelTransformer', 'ShapePropagator'])
