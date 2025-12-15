# Aquarium IDE Integration Tests - Quick Start Guide

## 🚀 Run Tests in 30 Seconds

```bash
# Install dependencies (if needed)
pip install -r requirements-dev.txt

# Run all Aquarium IDE tests
pytest tests/aquarium_ide/ -v

# Run with coverage report
pytest tests/aquarium_ide/ --cov=neural.aquarium --cov-report=html
open htmlcov/index.html
```

## 📋 Common Test Commands

### Run All Tests
```bash
pytest tests/aquarium_ide/ -v
```

### Run Specific Test File
```bash
pytest tests/aquarium_ide/test_integration.py -v
pytest tests/aquarium_ide/test_backend_api.py -v
pytest tests/aquarium_ide/test_welcome_components.py -v
pytest tests/aquarium_ide/test_e2e_workflows.py -v
pytest tests/aquarium_ide/test_real_examples.py -v
```

### Run Specific Test Class
```bash
pytest tests/aquarium_ide/test_integration.py::TestWelcomeScreen -v
pytest tests/aquarium_ide/test_backend_api.py::TestParseEndpoint -v
```

### Run Specific Test
```bash
pytest tests/aquarium_ide/test_integration.py::TestWelcomeScreen::test_welcome_screen_initialization -v
```

### Run by Marker
```bash
pytest tests/aquarium_ide/ -m welcome_screen -v
pytest tests/aquarium_ide/ -m templates -v
pytest tests/aquarium_ide/ -m examples -v
pytest tests/aquarium_ide/ -m api -v
pytest tests/aquarium_ide/ -m e2e -v
```

### Run with Different Verbosity
```bash
pytest tests/aquarium_ide/ -v          # Verbose
pytest tests/aquarium_ide/ -vv         # Very verbose
pytest tests/aquarium_ide/ -q          # Quiet
pytest tests/aquarium_ide/ --tb=short  # Short traceback
pytest tests/aquarium_ide/ --tb=no     # No traceback
```

### Run with Coverage
```bash
# HTML report
pytest tests/aquarium_ide/ --cov=neural.aquarium --cov-report=html

# Terminal report
pytest tests/aquarium_ide/ --cov=neural.aquarium --cov-report=term

# XML report (for CI/CD)
pytest tests/aquarium_ide/ --cov=neural.aquarium --cov-report=xml
```

### Run Failed Tests
```bash
# Run only failed tests from last run
pytest tests/aquarium_ide/ --lf

# Run failed tests first, then others
pytest tests/aquarium_ide/ --ff
```

### Stop on First Failure
```bash
pytest tests/aquarium_ide/ -x
pytest tests/aquarium_ide/ --maxfail=5  # Stop after 5 failures
```

### Run in Parallel
```bash
# Install pytest-xdist first: pip install pytest-xdist
pytest tests/aquarium_ide/ -n auto
pytest tests/aquarium_ide/ -n 4  # Use 4 workers
```

## 🎯 What Each Test File Does

| File | Purpose | Tests |
|------|---------|-------|
| `test_integration.py` | Core integration tests | 80+ |
| `test_backend_api.py` | FastAPI endpoint testing | 50+ |
| `test_welcome_components.py` | Frontend component tests | 90+ |
| `test_e2e_workflows.py` | End-to-end workflows | 70+ |
| `test_real_examples.py` | Real file validation | 60+ |

## 🔍 Quick Debugging

### See Print Statements
```bash
pytest tests/aquarium_ide/ -v -s
```

### Show Local Variables on Failure
```bash
pytest tests/aquarium_ide/ -v -l
```

### Start Debugger on Failure
```bash
pytest tests/aquarium_ide/ --pdb
```

### Show Which Tests Would Run
```bash
pytest tests/aquarium_ide/ --collect-only
```

## ✅ Pre-Commit Checklist

Before committing code, run:

```bash
# 1. Run all tests
pytest tests/aquarium_ide/ -v

# 2. Check coverage
pytest tests/aquarium_ide/ --cov=neural.aquarium --cov-report=term

# 3. Run linter
python -m ruff check neural/aquarium/

# 4. Run type checker
python -m mypy neural/aquarium/ --ignore-missing-imports
```

Or use the combined command:
```bash
pytest tests/aquarium_ide/ -v && \
python -m ruff check neural/aquarium/ && \
python -m mypy neural/aquarium/ --ignore-missing-imports
```

## 🐛 Troubleshooting

### Tests Can't Find Module
```bash
# Make sure you're in the repo root
cd /path/to/Neural

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Import Errors
```bash
# Install required dependencies
pip install fastapi
pip install pytest pytest-cov

# Or install all dev dependencies
pip install -r requirements-dev.txt
```

### Test Discovery Issues
```bash
# Check if tests are discovered
pytest tests/aquarium_ide/ --collect-only

# Make sure __init__.py exists
ls tests/aquarium_ide/__init__.py
```

### Fixture Not Found
```bash
# Check conftest.py exists
ls tests/aquarium_ide/conftest.py
ls tests/conftest.py

# Run with fixture list
pytest tests/aquarium_ide/ --fixtures
```

## 📊 Test Coverage Goals

| Component | Current | Target |
|-----------|---------|--------|
| Welcome Screen | 100% | 100% ✅ |
| Templates | 100% | 100% ✅ |
| Example Gallery | 100% | 100% ✅ |
| DSL Compilation | 100% | 100% ✅ |
| Backend API | 100% | 100% ✅ |
| Shape Propagation | 95% | 90% ✅ |
| Error Handling | 100% | 100% ✅ |

## 🎓 Writing New Tests

### 1. Choose the Right File
- **Integration tests** → `test_integration.py`
- **API tests** → `test_backend_api.py`
- **UI component tests** → `test_welcome_components.py`
- **Workflow tests** → `test_e2e_workflows.py`
- **File validation** → `test_real_examples.py`

### 2. Use Existing Fixtures
```python
def test_my_feature(sample_templates, example_files):
    # Use fixtures from conftest.py
    template = sample_templates[0]
    # ... your test code
```

### 3. Follow Naming Convention
```python
class TestMyFeature:
    """Test suite for my feature."""
    
    def test_feature_success(self):
        """Test successful operation."""
        pass
    
    def test_feature_error_handling(self):
        """Test error handling."""
        pass
```

### 4. Add Markers
```python
@pytest.mark.welcome_screen
def test_welcome_feature():
    pass
```

### 5. Document Your Test
```python
def test_something_important(self):
    """
    Test that something important works.
    
    This test verifies that:
    - Feature X initializes correctly
    - Error Y is handled properly
    - Result Z is as expected
    """
    # Test code
```

## 🚦 CI/CD Integration

### GitHub Actions Example
```yaml
name: Aquarium IDE Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          pip install -e .
      - name: Run tests
        run: |
          pytest tests/aquarium_ide/ -v --tb=short
          pytest tests/aquarium_ide/ --cov=neural.aquarium --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## 📚 Additional Resources

- **Full Documentation**: See `README.md` in this directory
- **Test Summary**: See `INTEGRATION_TEST_SUMMARY.md`
- **Fixtures Reference**: See `conftest.py`
- **Test Markers**: Run `pytest --markers`

## 💡 Tips & Best Practices

1. **Run tests frequently** - Catch issues early
2. **Write tests first** - Test-driven development
3. **Keep tests fast** - Use mocking when appropriate
4. **Test edge cases** - Don't just test the happy path
5. **Update tests** - When fixing bugs, add regression tests
6. **Check coverage** - Aim for 90%+ coverage
7. **Document tests** - Clear docstrings help future developers

## 🎉 Success Criteria

Your code is ready when:
- ✅ All tests pass
- ✅ Coverage >= 90%
- ✅ No linting errors
- ✅ Type checking passes
- ✅ Documentation updated

## 📞 Need Help?

1. Check the `README.md` for detailed documentation
2. Look at existing tests for examples
3. Check fixtures in `conftest.py`
4. Review test output carefully
5. Use `pytest --fixtures` to see available fixtures

---

**Quick Links:**
- [Full README](README.md)
- [Test Summary](INTEGRATION_TEST_SUMMARY.md)
- [Test Configuration](conftest.py)

Happy Testing! 🧪
