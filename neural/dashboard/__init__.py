"""
Neural Dashboard Module
Provides debugging and visualization tools for neural networks.
"""

# Lazy imports to avoid blocking on module load
app = None
server = None
DEBUGGER_AVAILABLE = False
create_debugger_backend = None
setup_debugger_routes = None
DebuggerBackend = None


def _load_dashboard():
    """Lazy load dashboard components."""
    global app, server
    if app is None:
        try:
            from neural.dashboard.dashboard import app as _app
            from neural.dashboard.dashboard import server as _server
            app = _app
            server = _server
        except ImportError as e:
            raise ImportError(f"Dashboard dependencies not installed: {e}")
    return app, server


def _load_debugger():
    """Lazy load debugger components."""
    global DEBUGGER_AVAILABLE, create_debugger_backend, setup_debugger_routes, DebuggerBackend
    if not DEBUGGER_AVAILABLE:
        try:
            from neural.dashboard.debugger_backend import (
                DebuggerBackend as _Backend,
            )
            from neural.dashboard.debugger_backend import (
                create_debugger_backend as _create,
            )
            from neural.dashboard.debugger_backend import (
                setup_debugger_routes as _setup,
            )
            create_debugger_backend = _create
            setup_debugger_routes = _setup
            DebuggerBackend = _Backend
            DEBUGGER_AVAILABLE = True
        except ImportError:
            pass
    return DEBUGGER_AVAILABLE


__all__ = [
    'app',
    'server',
    'create_debugger_backend',
    'setup_debugger_routes',
    'DebuggerBackend',
    'DEBUGGER_AVAILABLE',
    '_load_dashboard',
    '_load_debugger',
]
