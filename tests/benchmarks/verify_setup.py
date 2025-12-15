#!/usr/bin/env python
"""
Verify that the benchmarking suite is properly set up.

This script checks:
1. All required modules can be imported
2. Model definitions are valid
3. Basic benchmark functionality works
"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_imports():
    """Check that all required modules can be imported."""
    print("Checking imports...")
    
    try:
        from tests.benchmarks import BenchmarkRunner, BenchmarkSuite, get_benchmark_models
        print("  ✓ BenchmarkRunner imported")
        print("  ✓ BenchmarkSuite imported")
        print("  ✓ get_benchmark_models imported")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    
    try:
        from neural.parser.parser import ModelTransformer, create_parser
        print("  ✓ Neural parser imported")
    except ImportError as e:
        print(f"  ✗ Neural parser import error: {e}")
        return False
    
    try:
        from neural.code_generation.code_generator import generate_code
        print("  ✓ Code generator imported")
    except ImportError as e:
        print(f"  ✗ Code generator import error: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__} available")
    except ImportError:
        print("  ⚠ TensorFlow not available (optional)")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__} available")
    except ImportError:
        print("  ⚠ PyTorch not available (optional)")
    
    try:
        import psutil
        print("  ✓ psutil available")
    except ImportError as e:
        print(f"  ✗ psutil required but not installed: {e}")
        print("    Install with: pip install psutil")
        return False
    
    return True


def check_models():
    """Check that model definitions are valid."""
    print("\nChecking model definitions...")
    
    try:
        from tests.benchmarks import get_benchmark_models
        models = get_benchmark_models()
        
        required_models = ['simple_mlp', 'cnn', 'deep_mlp']
        for model_name in required_models:
            if model_name not in models:
                print(f"  ✗ Model '{model_name}' not found")
                return False
            
            model = models[model_name]
            if 'neural_dsl' not in model:
                print(f"  ✗ Model '{model_name}' missing neural_dsl definition")
                return False
            if 'tensorflow' not in model:
                print(f"  ✗ Model '{model_name}' missing tensorflow definition")
                return False
            if 'pytorch' not in model:
                print(f"  ✗ Model '{model_name}' missing pytorch definition")
                return False
            
            print(f"  ✓ Model '{model_name}' valid")
        
        print(f"  ✓ All {len(required_models)} models valid")
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking models: {e}")
        return False


def check_dsl_parsing():
    """Check that DSL models can be parsed."""
    print("\nChecking DSL parsing...")
    
    try:
        from tests.benchmarks import get_benchmark_models

        from neural.parser.parser import ModelTransformer, create_parser
        
        models = get_benchmark_models()
        parser = create_parser('network')
        
        for model_name, model_def in models.items():
            try:
                tree = parser.parse(model_def['neural_dsl'])
                ModelTransformer().transform(tree)
                print(f"  ✓ {model_name} DSL parsed successfully")
            except Exception as e:
                print(f"  ✗ {model_name} DSL parsing failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during DSL parsing: {e}")
        return False


def check_code_generation():
    """Check that code can be generated from DSL."""
    print("\nChecking code generation...")
    
    try:
        from tests.benchmarks import get_benchmark_models

        from neural.code_generation.code_generator import generate_code
        from neural.parser.parser import ModelTransformer, create_parser
        
        models = get_benchmark_models()
        parser = create_parser('network')
        
        model_def = models['simple_mlp']
        tree = parser.parse(model_def['neural_dsl'])
        model_data = ModelTransformer().transform(tree)
        
        for backend in ['tensorflow', 'pytorch']:
            try:
                code = generate_code(model_data, backend)
                if len(code) > 0:
                    print(f"  ✓ Code generated for {backend}")
                else:
                    print(f"  ✗ Empty code generated for {backend}")
                    return False
            except Exception as e:
                print(f"  ✗ Code generation failed for {backend}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during code generation: {e}")
        return False


def check_benchmark_runner():
    """Check that BenchmarkRunner can be instantiated."""
    print("\nChecking BenchmarkRunner...")
    
    try:
        from tests.benchmarks import BenchmarkRunner
        
        runner = BenchmarkRunner()
        print("  ✓ BenchmarkRunner created")
        print(f"  ✓ Output directory: {runner.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ BenchmarkRunner error: {e}")
        return False


def check_benchmark_suite():
    """Check that BenchmarkSuite can be instantiated."""
    print("\nChecking BenchmarkSuite...")
    
    try:
        from tests.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        print("  ✓ BenchmarkSuite created")
        print(f"  ✓ Output directory: {suite.runner.output_dir}")
        print(f"  ✓ Models available: {len(suite.models)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ BenchmarkSuite error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("Neural DSL Benchmarking Suite Verification")
    print("=" * 80)
    
    checks = [
        ("Imports", check_imports),
        ("Model definitions", check_models),
        ("DSL parsing", check_dsl_parsing),
        ("Code generation", check_code_generation),
        ("BenchmarkRunner", check_benchmark_runner),
        ("BenchmarkSuite", check_benchmark_suite),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results[name] = False
    
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All checks passed! Benchmarking suite is ready to use.")
        print("\nNext steps:")
        print("  1. Run quick benchmark: python tests/benchmarks/run_benchmarks.py --quick")
        print("  2. Run full benchmark: python tests/benchmarks/run_benchmarks.py --report")
        print("  3. View examples: python tests/benchmarks/example_usage.py")
        return 0
    else:
        print("✗ Some checks failed. Please fix the errors above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -e '.[full]'")
        print("  - Check Python version: python --version (requires 3.8+)")
        print("  - Verify Neural DSL installation: pip show neural-dsl")
        return 1


if __name__ == '__main__':
    sys.exit(main())
