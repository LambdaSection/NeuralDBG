#!/usr/bin/env python
"""
Test runner script to execute all tests and capture results.
"""
import sys
import os
import subprocess
import json

def run_all_tests():
    """Run all tests in the tests directory."""
    print("Running comprehensive test suite for Neural framework...")

    # Change to the Neural directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Add current directory to Python path
    sys.path.insert(0, '.')

    # Try to run pytest on all test files
    try:
        import pytest
        import pytest_json

        # Configure test run
        test_dir = 'tests'

        print(f"Running tests in {test_dir}/...")

        # Run pytest with JSON output
        result = pytest.main([
            test_dir,
            '--tb=short',
            '--maxfail=10',
            '-v',
            '--durations=10',
            '-x',  # Stop on first failure
        ])

        if result == 0:
            print("All tests passed!")
        elif result == 1:
            print("Some tests failed")
        else:
            print(f"Test execution completed with code {result}")

    except ImportError as e:
        print(f"Cannot run pytest: {e}")
        print("Checking if tests can run manually...")
        run_manual_tests()

def run_manual_tests():
    """Run basic tests manually if pytest is not available."""
    print("Running manual test verification...")

    results = []

    # Initialize variables for scope
    ShapePropagator = None
    create_parser = None
    cli = None

    # Test 1: Import check
    try:
        from neural.parser.parser import create_parser
        from neural.shape_propagation.shape_propagator import ShapePropagator
        from neural.cli import cli
        results.append(("Core Imports", True, "All core modules imported successfully"))
        print("Core imports test passed")
    except Exception as e:
        results.append(("Core Imports", False, str(e)))
        print(f"Core imports test failed: {e}")

    # Test 2: Shape propagation (only if imports worked)
    if ShapePropagator is not None:
        try:
            propagator = ShapePropagator()
            input_shape = (1, 28, 28, 3)
            layer = {
                'type': 'Conv2D',
                'params': {
                    'filters': 16,
                    'kernel_size': 3,
                    'padding': 1,
                    'stride': 1
                }
            }
            output_shape = propagator.propagate(input_shape, layer, 'tensorflow')
            expected = (1, 28, 28, 16)
            if output_shape == expected:
                results.append(("Shape Propagation", True, f"Shape calculation correct: {input_shape} -> {output_shape}"))
                print("Shape propagation test passed")
            else:
                results.append(("Shape Propagation", False, f"Wrong output: got {output_shape}, expected {expected}"))
                print(f"Shape propagation test failed: got {output_shape}, expected {expected}")
        except Exception as e:
            results.append(("Shape Propagation", False, str(e)))
            print(f"Shape propagation test failed: {e}")
    else:
        results.append(("Shape Propagation", False, "Import failed"))
        print("Shape propagation test skipped: import failed")

    # Test 3: Parser functionality (only if imports worked)
    if create_parser is not None:
        try:
            p = create_parser()
            test_content = "network Test { input: (10,) layers: Dense(units=5) }"
            result = p.parse(test_content)
            results.append(("DSL Parser", True, "Successfully parsed simple network"))
            print("DSL parser test passed")
        except Exception as e:
            results.append(("DSL Parser", False, str(e)))
            print(f"DSL parser test failed: {e}")
    else:
        results.append(("DSL Parser", False, "Import failed"))
        print("DSL parser test skipped: import failed")

    # Test 4: Compilation check
    try:
        # Test if we can execute the CLI
        result = subprocess.run([
            sys.executable, '-m', 'neural', 'compile', '--help'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            results.append(("CLI Functionality", True, "CLI help command executed successfully"))
            print("CLI functionality test passed")
        else:
            results.append(("CLI Functionality", False, f"CLI failed: {result.stderr}"))
            print(f"CLI functionality test failed: {result.stderr}")
    except Exception as e:
        results.append(("CLI Functionality", False, str(e)))
        print(f"CLI functionality test failed: {e}")

    # Summary
    print("\nTest Results Summary:")
    passed = 0
    for test_name, success, details in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} {test_name}: {details}")
        if success:
            passed += 1

    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("All manual tests passed! Neural framework is working correctly.")
    elif passed >= total * 0.8:
        print("Most tests passed. Minor issues detected but core functionality works.")
    else:
        print("Major issues detected. Framework needs attention.")

if __name__ == "__main__":
    run_all_tests()
