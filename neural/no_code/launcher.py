"""
Launcher for the No-Code Interface
Supports both classic Dash UI and modern React Flow UI
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description='Launch Neural DSL No-Code Interface')
    parser.add_argument(
        '--ui',
        choices=['react', 'dash', 'classic'],
        default='react',
        help='UI type to launch (react: modern React Flow UI, dash/classic: original Dash UI)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8051,
        help='Port to run the server on (default: 8051)'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    if args.ui == 'react':
        print("🚀 Starting Neural DSL No-Code Designer (React Flow UI)")
        print(f"🌐 Open your browser to http://{args.host}:{args.port}")
        from neural.no_code.app import app
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        print("🚀 Starting Neural DSL No-Code Interface (Classic Dash UI)")
        print(f"🌐 Open your browser to http://{args.host}:{args.port}")
        from neural.no_code.no_code import app
        app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
