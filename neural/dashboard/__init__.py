"""
Neural Dashboard Module
Provides debugging and visualization tools for neural networks.
"""

from neural.dashboard.dashboard import app, server

try:
    from neural.dashboard.debugger_backend import (
        create_debugger_backend,
        setup_debugger_routes,
        DebuggerBackend,
    )
    DEBUGGER_AVAILABLE = True
except ImportError:
    DEBUGGER_AVAILABLE = False
    create_debugger_backend = None
    setup_debugger_routes = None
    DebuggerBackend = None

__all__ = [
    'app',
    'server',
    'create_debugger_backend',
    'setup_debugger_routes',
    'DebuggerBackend',
    'DEBUGGER_AVAILABLE',
]
