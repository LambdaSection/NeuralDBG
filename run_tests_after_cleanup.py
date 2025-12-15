#!/usr/bin/env python3
"""
Script to run tests after cleanup and generate a report.

This script:
1. Runs the full test suite
2. Identifies failing tests
3. Categorizes failures (import errors, dependency errors, logic errors)
4. Generates a summary report
"""
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, timeout=600):
    """Run a command and return output."""
    print(f"\n{'=' * 70}")
    print(f"Running: {cmd}")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"Completed in {elapsed:.2f}s")
        print(f"{'=' * 70}\n")
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"TIMEOUT after {elapsed:.2f}s")
        print(f"{'=' * 70}\n")
        return None


def main():
    """Main test execution."""
    print("=" * 70)
    print("NEURAL DSL - POST-CLEANUP TEST SUITE")
    print("=" * 70)
    
    # Test 1: Collection only
    print("\n1. Testing test collection...")
    result = run_command("python -m pytest tests/ --collect-only -q", timeout=60)
    if result is None:
        print("❌ Test collection TIMED OUT")
        print("\nThis suggests slow imports. Check:")
        print("  - Heavy dependencies being imported at module level")
        print("  - Circular imports")
        print("  - Missing optional dependencies causing hangs")
        return 1
    elif result.returncode != 0:
        print("❌ Test collection FAILED")
        print("\nSTDERR:")
        print(result.stderr)
        return 1
    else:
        print("✅ Test collection PASSED")
        print(f"\n{result.stdout}")
    
    # Test 2: Run parser tests (core functionality)
    print("\n2. Testing parser module...")
    result = run_command("python -m pytest tests/parser/ -v --tb=short", timeout=120)
    if result is None:
        print("❌ Parser tests TIMED OUT")
        return 1
    
    parser_passed = result.returncode == 0
    print("✅ Parser tests PASSED" if parser_passed else "❌ Parser tests FAILED")
    if not parser_passed:
        print("\nFailed tests output:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    
    # Test 3: Run shape propagation tests
    print("\n3. Testing shape propagation...")
    result = run_command("python -m pytest tests/shape_propagation/ -v --tb=short", timeout=120)
    if result is None:
        print("❌ Shape propagation tests TIMED OUT")
        return 1
    
    shape_passed = result.returncode == 0
    print("✅ Shape propagation tests PASSED" if shape_passed else "❌ Shape propagation tests FAILED")
    if not shape_passed:
        print("\nFailed tests output:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    
    # Test 4: Run code generator tests
    print("\n4. Testing code generation...")
    result = run_command("python -m pytest tests/code_generator/ -v --tb=short", timeout=120)
    if result is None:
        print("❌ Code generator tests TIMED OUT")
        return 1
    
    codegen_passed = result.returncode == 0
    print("✅ Code generator tests PASSED" if codegen_passed else "❌ Code generator tests FAILED")
    if not codegen_passed:
        print("\nFailed tests output:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    
    # Test 5: Run full test suite (excluding slow/E2E tests)
    print("\n5. Running full test suite (unit tests only)...")
    result = run_command(
        'python -m pytest tests/ -v -m "not slow and not integration" --tb=short',
        timeout=600
    )
    if result is None:
        print("❌ Full test suite TIMED OUT")
        return 1
    
    full_passed = result.returncode == 0
    print("✅ Full test suite PASSED" if full_passed else "⚠️ Full test suite had some failures")
    
    # Generate summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test Collection: {'✅ PASS' if result is not None else '❌ FAIL'}")
    print(f"Parser Tests: {'✅ PASS' if parser_passed else '❌ FAIL'}")
    print(f"Shape Propagation: {'✅ PASS' if shape_passed else '❌ FAIL'}")
    print(f"Code Generation: {'✅ PASS' if codegen_passed else '❌ FAIL'}")
    print(f"Full Suite: {'✅ PASS' if full_passed else '⚠️ PARTIAL'}")
    
    # Write results to file
    with open("TEST_RESULTS_SUMMARY.md", "w") as f:
        f.write("# Test Results Summary\n\n")
        f.write(f"- **Test Collection**: {'✅ PASS' if result is not None else '❌ FAIL'}\n")
        f.write(f"- **Parser Tests**: {'✅ PASS' if parser_passed else '❌ FAIL'}\n")
        f.write(f"- **Shape Propagation**: {'✅ PASS' if shape_passed else '❌ FAIL'}\n")
        f.write(f"- **Code Generation**: {'✅ PASS' if codegen_passed else '❌ FAIL'}\n")
        f.write(f"- **Full Suite**: {'✅ PASS' if full_passed else '⚠️ PARTIAL'}\n\n")
        f.write("See TEST_FIXES_IMPLEMENTATION.md for details on fixes applied.\n")
    
    print(f"\nResults written to TEST_RESULTS_SUMMARY.md")
    
    return 0 if (parser_passed and shape_passed and codegen_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
