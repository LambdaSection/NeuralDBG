"""
Neural Aquarium - AI-Powered Neural DSL Builder

Web-based interface for creating Neural DSL models through natural language conversation.
Includes integrated debugger with NeuralDbg dashboard embedding.
"""

__version__ = "0.3.0"
__author__ = "Neural DSL Team"

from pathlib import Path

AQUARIUM_ROOT = Path(__file__).parent
FRONTEND_ROOT = AQUARIUM_ROOT / "src"
BACKEND_ROOT = AQUARIUM_ROOT / "backend"

__all__ = [
    "__version__",
    "__author__",
    "AQUARIUM_ROOT",
    "FRONTEND_ROOT",
    "BACKEND_ROOT",
]
