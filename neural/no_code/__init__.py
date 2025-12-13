"""
No-code interface for Neural DSL
Provides both classic Dash UI and modern React Flow UI
"""

from .app import app as flask_app
from .no_code import app as dash_app

__all__ = ['flask_app', 'dash_app']
