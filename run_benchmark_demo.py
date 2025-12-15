#!/usr/bin/env python
"""
Quick benchmark demo launcher.

This is the absolute easiest way to see Neural DSL benchmarks in action.
Just run this script from the project root!

Usage:
    python run_benchmark_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run the quick start benchmark demo."""
    print("\n" + "="*70)
    print(" "*20 + "NEURAL DSL BENCHMARK DEMO")
    print("="*70)
    print("\nThis demo will:")
    print("  • Compare Neural DSL vs Keras")
    print("  • Show code reduction (60-75%)")
    print("  • Demonstrate development speed (3-5x faster)")
    print("  • Validate model accuracy (equivalent)")
    print("  • Display results in ~2 minutes")
    print("\n" + "="*70 + "\n")
    
    try:
        from neural.benchmarks import quick_start
        quick_start.main()
    except ImportError as e:
        print(f"❌ Error: Missing dependencies\n")
        print(f"Details: {e}\n")
        print("To fix:")
        print("  1. Install Neural DSL:")
        print("     pip install -e \".[full]\"")
        print("\n  2. Install benchmark dependencies:")
        print("     pip install -r neural/benchmarks/requirements.txt")
        print("\n  3. Run again:")
        print("     python run_benchmark_demo.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFor help:")
        print("  • See neural/benchmarks/README.md")
        print("  • Open an issue: https://github.com/your-org/neural-dsl/issues")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
