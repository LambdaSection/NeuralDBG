#!/usr/bin/env python
"""
HPO API Server Launcher
Starts the Flask server for HPO execution and monitoring
"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.aquarium.api.hpo_api import run_server


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='HPO API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5003, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔬 Neural Aquarium - HPO API Server")
    print("="*70)
    print(f"\n🚀 Starting HPO server on http://{args.host}:{args.port}")
    print(f"   Debug Mode: {args.debug}")
    print("\n📝 Features:")
    print("   • HPO Configuration and Execution")
    print("   • Real-time Trial Streaming")
    print("   • Optuna Backend Integration")
    print("   • Multi-backend Support (PyTorch, TensorFlow)")
    print(f"\n🌐 API Endpoints available at: http://localhost:{args.port}/api/hpo/")
    print("   Press Ctrl+C to stop the server\n")
    print("="*70)
    
    run_server(host=args.host, port=args.port, debug=args.debug)
