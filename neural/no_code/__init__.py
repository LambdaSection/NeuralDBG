"""
No-code interface for Neural DSL
Provides both classic Dash UI and modern React Flow UI
"""

# Lazy imports to avoid blocking on module load
flask_app = None
dash_app = None


def _load_apps():
    """Lazy load no-code apps."""
    global flask_app, dash_app
    if flask_app is None:
        try:
            from .app import app as _flask_app
            from .no_code import app as _dash_app
            flask_app = _flask_app
            dash_app = _dash_app
        except ImportError as e:
            raise ImportError(f"No-code dependencies not installed: {e}")
    return flask_app, dash_app


__all__ = ['flask_app', 'dash_app', '_load_apps']
