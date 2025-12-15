"""
Enhanced NeuralDbg dashboard with integrated debugger support.
Extends the base dashboard.py with debugger backend functionality.
"""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.dashboard.dashboard import app, server, socketio
from neural.dashboard.debugger_backend import (
    SOCKETIO_AVAILABLE,
    create_debugger_backend,
    setup_debugger_routes,
)


debugger_backend = None

if SOCKETIO_AVAILABLE and socketio is not None:
    debugger_backend = create_debugger_backend(socketio)
    setup_debugger_routes(server, socketio, debugger_backend)
    print("Debugger backend initialized")
else:
    print("Warning: flask_socketio not available, debugger backend disabled")


def get_debugger():
    """Get the debugger backend instance."""
    return debugger_backend


if __name__ == "__main__":
    if socketio and SOCKETIO_AVAILABLE:
        socketio.run(server, host="localhost", port=8050, debug=False, allow_unsafe_werkzeug=True)
    else:
        app.run_server(host="localhost", port=8050, debug=False, use_reloader=False)
