"""
Neural Aquarium - AI-Powered Neural DSL Builder

Web-based interface for creating Neural DSL models through natural language conversation.
"""

__version__ = "0.1.0"
__author__ = "Neural DSL Team"

from pathlib import Path

AQUARIUM_ROOT = Path(__file__).parent
FRONTEND_ROOT = AQUARIUM_ROOT / "src"
BACKEND_ROOT = AQUARIUM_ROOT / "backend"
