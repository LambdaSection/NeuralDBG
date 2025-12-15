#!/bin/bash
# Script to run Aquarium E2E tests on Unix-like systems

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "======================================================================"
echo "Aquarium IDE End-to-End Tests"
echo "======================================================================"

# Check if Playwright is installed
if ! python -c "import playwright" 2>/dev/null; then
    echo "Error: Playwright is not installed"
    echo "Install with: pip install playwright && playwright install chromium"
    exit 1
fi

# Check if Aquarium server dependencies are available
if ! python -c "import dash" 2>/dev/null; then
    echo "Warning: Dash not installed. Some tests may fail."
    echo "Install with: pip install -e .[dashboard]"
fi

# Run tests
cd "$SCRIPT_DIR"

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./run_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --fast        Skip slow tests"
    echo "  --visible     Run with visible browser"
    echo "  --debug       Run in debug mode"
    echo "  --parallel    Run tests in parallel"
    echo "  --help        Show this help message"
    exit 0
fi

python run_tests.py "$@"
