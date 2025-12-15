#!/usr/bin/env python3
"""Quick script to check if core imports work after cleanup fixes."""
import sys
import time


def time_import(module_path, description=""):
    """Time an import and report results."""
    print(f"\nTesting: {description or module_path}...", end=" ", flush=True)
    start = time.time()
    try:
        parts = module_path.split(".")
        module = __import__(module_path)
        for part in parts[1:]:
            module = getattr(module, part)
        elapsed = time.time() - start
        print(f"✅ {elapsed:.3f}s")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ {elapsed:.3f}s")
        print(f"   Error: {e}")
        return False, elapsed


def main():
    """Run import checks."""
    print("=" * 70)
    print("IMPORT HEALTH CHECK")
    print("=" * 70)
    
    results = []
    
    # Check core dependencies
    print("\n📦 Core Dependencies:")
    results.append(time_import("pytest", "pytest"))
    results.append(time_import("numpy", "numpy"))
    results.append(time_import("lark", "lark"))
    results.append(time_import("click", "click"))
    
    # Check neural modules
    print("\n🧠 Neural DSL Modules:")
    results.append(time_import("neural", "neural (main package)"))
    results.append(time_import("neural.exceptions", "exceptions"))
    results.append(time_import("neural.parser.parser", "parser"))
    results.append(time_import("neural.shape_propagation.shape_propagator", "shape propagator"))
    results.append(time_import("neural.code_generation.code_generator", "code generator"))
    
    # Check optional modules
    print("\n🔧 Optional Modules:")
    results.append(time_import("neural.utils", "utils"))
    results.append(time_import("neural.cli", "cli"))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_imports = len(results)
    successful = sum(1 for success, _ in results if success)
    failed = total_imports - successful
    total_time = sum(elapsed for _, elapsed in results)
    
    print(f"\nTotal imports tested: {total_imports}")
    print(f"Successful: {successful} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Total import time: {total_time:.3f}s")
    print(f"Average import time: {total_time/total_imports:.3f}s")
    
    if failed > 0:
        print("\n⚠️ Some imports failed. Check dependencies or module issues.")
        return 1
    elif total_time > 10:
        print("\n⚠️ Import time is slow (>10s). Consider lazy loading more modules.")
        return 1
    else:
        print("\n✅ All imports successful and performant!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
