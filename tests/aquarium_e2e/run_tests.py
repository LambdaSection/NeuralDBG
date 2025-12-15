#!/usr/bin/env python
"""
Script to run Aquarium E2E tests with various configurations.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --visible          # Run with visible browser
    python run_tests.py --debug            # Run in debug mode
    python run_tests.py --file test_dsl_editor.py  # Run specific file
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_tests(
    test_path: str = ".",
    fast: bool = False,
    visible: bool = False,
    debug: bool = False,
    parallel: bool = False,
    verbose: bool = True,
    slow_mo: int = 0,
    extra_args: list[str] = None
):
    """
    Run E2E tests with specified configuration.
    
    Args:
        test_path: Path to test file or directory
        fast: Skip slow tests
        visible: Run with visible browser (not headless)
        debug: Enable debug mode with Playwright inspector
        parallel: Run tests in parallel
        verbose: Enable verbose output
        slow_mo: Slow down operations by N milliseconds
        extra_args: Additional pytest arguments
    """
    env = os.environ.copy()
    
    if visible:
        env['HEADLESS'] = 'false'
    
    if slow_mo > 0:
        env['SLOW_MO'] = str(slow_mo)
    
    if debug:
        env['PWDEBUG'] = '1'
        env['HEADLESS'] = 'false'
    
    pytest_args = [
        sys.executable, '-m', 'pytest',
        str(Path(__file__).parent / test_path)
    ]
    
    if verbose:
        pytest_args.append('-v')
    
    if fast:
        pytest_args.extend(['-m', 'not slow'])
    
    if parallel and not debug:
        pytest_args.extend(['-n', 'auto'])
    
    pytest_args.append('--tb=short')
    
    if extra_args:
        pytest_args.extend(extra_args)
    
    print("=" * 70)
    print("Running Aquarium IDE End-to-End Tests")
    print("=" * 70)
    print(f"Test path: {test_path}")
    print(f"Headless: {not visible}")
    print(f"Debug mode: {debug}")
    print(f"Parallel: {parallel}")
    print(f"Slow motion: {slow_mo}ms")
    print("=" * 70)
    print()
    
    try:
        result = subprocess.run(pytest_args, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nError running tests: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run Aquarium IDE E2E tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                           # Run all tests
  python run_tests.py --fast                    # Skip slow tests
  python run_tests.py --visible                 # Run with visible browser
  python run_tests.py --debug                   # Debug mode with inspector
  python run_tests.py --file test_dsl_editor.py # Run specific file
  python run_tests.py --parallel                # Run tests in parallel
  python run_tests.py --visible --slow-mo 500   # Visual + slow motion
        """
    )
    
    parser.add_argument(
        '--file',
        type=str,
        default='.',
        help='Specific test file to run (default: all tests)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Skip slow tests'
    )
    
    parser.add_argument(
        '--visible',
        action='store_true',
        help='Run with visible browser (not headless)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with Playwright inspector'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    
    parser.add_argument(
        '--slow-mo',
        type=int,
        default=0,
        help='Slow down operations by N milliseconds'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        'pytest_args',
        nargs='*',
        help='Additional arguments to pass to pytest'
    )
    
    args = parser.parse_args()
    
    return_code = run_tests(
        test_path=args.file,
        fast=args.fast,
        visible=args.visible,
        debug=args.debug,
        parallel=args.parallel,
        verbose=not args.quiet,
        slow_mo=args.slow_mo,
        extra_args=args.pytest_args
    )
    
    sys.exit(return_code)


if __name__ == '__main__':
    main()
