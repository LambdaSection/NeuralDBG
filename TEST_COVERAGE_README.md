# Neural DSL - Test Coverage System

This document describes how to run tests with coverage analysis and generate coverage reports.

## Quick Start

### Option 1: Using the Script (Recommended)

**Windows:**
```bash
python generate_test_coverage_summary.py
```

**Unix/Linux/macOS:**
```bash
python generate_test_coverage_summary.py
```

### Option 2: Using Convenience Scripts

**Windows:**
```bash
.\run_tests_with_coverage.bat
```

**Unix/Linux/macOS:**
```bash
chmod +x run_tests_with_coverage.sh
./run_tests_with_coverage.sh
```

### Option 3: Using Make

```bash
make test-cov-report
```

Or just run tests with coverage:
```bash
make test-cov
```

### Option 4: Direct pytest Command

```bash
pytest tests/ -v --cov=neural --cov-report=term --cov-report=html
```

## What Gets Generated

Running the coverage script generates:

1. **TEST_COVERAGE_SUMMARY.md** - Executive summary with:
   - Test statistics (passed, failed, skipped)
   - Overall coverage percentage
   - Per-module coverage breakdown
   - Recommendations for improvements
   - Comparison with previous run (if available)

2. **htmlcov/index.html** - Interactive HTML coverage report
   - Browse coverage by file
   - See line-by-line coverage highlighting
   - Identify uncovered code paths

3. **coverage.json** - Machine-readable coverage data
   - Used by CI/CD systems
   - Parseable for custom analysis

## Configuration

### pytest Configuration

Test configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra", "--strict-markers", "--strict-config", "--showlocals"]
```

### Coverage Configuration

Coverage settings are also in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["neural"]
branch = true
omit = ["*/tests/*", "*/test_*.py", "*/__pycache__/*"]

[tool.coverage.report]
precision = 2
show_missing = true
exclude_lines = ["pragma: no cover", "if __name__ == .__main__.:", ...]
```

## Test Markers

Tests can be marked with custom markers for selective running:

- `@pytest.mark.slow` - Slow tests (skip with `-m "not slow"`)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.requires_gpu` - Requires GPU
- `@pytest.mark.requires_torch` - Requires PyTorch
- `@pytest.mark.requires_tensorflow` - Requires TensorFlow
- `@pytest.mark.requires_onnx` - Requires ONNX

### Examples

Run only fast tests:
```bash
pytest tests/ -v -m "not slow"
```

Run only unit tests:
```bash
pytest tests/ -v -m unit
```

Run tests that don't require TensorFlow:
```bash
pytest tests/ -v -m "not requires_tensorflow"
```

## Viewing Coverage Reports

### Terminal Output

Coverage summary is printed to terminal automatically when using `--cov-report=term`.

### HTML Report

Open the interactive HTML report:

**Windows:**
```bash
start htmlcov/index.html
```

**macOS:**
```bash
open htmlcov/index.html
```

**Linux:**
```bash
xdg-open htmlcov/index.html
```

### Markdown Summary

Read the executive summary:
```bash
cat TEST_COVERAGE_SUMMARY.md
```

Or open in your favorite markdown viewer.

## CI/CD Integration

### GitHub Actions

The repository includes CI workflows that run tests with coverage. See `.github/workflows/ci.yml`.

### Coverage Tracking

The `TEST_COVERAGE_SUMMARY.md` file tracks improvements over time by comparing with the previous run.

## Troubleshooting

### Missing Dependencies

If you get import errors, ensure development dependencies are installed:

```bash
pip install -r requirements-dev.txt
```

Or install with development extras:

```bash
pip install -e ".[full]"
pip install pytest pytest-cov
```

### No Coverage Data

If no coverage data is generated:

1. Check that `pytest-cov` is installed: `pip install pytest-cov`
2. Ensure tests are running: `pytest tests/ -v`
3. Check that the `neural/` directory exists and contains Python files

### Tests Not Found

If pytest can't find tests:

1. Ensure you're in the repository root directory
2. Check that test files follow the naming convention: `test_*.py` or `*_test.py`
3. Verify test functions start with `test_`

### Import Errors in Tests

If tests fail with import errors:

1. Install the package in development mode: `pip install -e .`
2. Install optional dependencies if needed: `pip install -e ".[full]"`
3. Check the test file imports - they should use absolute imports from `neural.*`

## Best Practices

### Writing Tests

1. **Use descriptive test names**: `test_parser_handles_invalid_syntax`
2. **One assertion per test** (when possible): Makes failures easier to diagnose
3. **Use fixtures**: Share setup code between tests
4. **Mark slow tests**: Use `@pytest.mark.slow` for tests that take >1 second
5. **Mark dependencies**: Use appropriate markers for optional dependencies

### Coverage Goals

- **Minimum**: 80% overall coverage
- **Target**: 90% overall coverage
- **Critical modules** (parser, code_generation): 95%+ coverage
- **New code**: 100% coverage for new features

### Running Tests During Development

For fast feedback during development:

```bash
# Run specific test file
pytest tests/parser/test_parser.py -v

# Run specific test function
pytest tests/parser/test_parser.py::test_parse_simple_model -v

# Run with coverage for specific module
pytest tests/parser/ -v --cov=neural.parser --cov-report=term

# Stop on first failure
pytest tests/ -v -x

# Show local variables on failure
pytest tests/ -v --showlocals

# Run only failed tests from last run
pytest tests/ -v --lf
```

## Script Reference

### generate_test_coverage_summary.py

Main script that:
1. Runs pytest with coverage options
2. Parses test results and coverage data
3. Generates `TEST_COVERAGE_SUMMARY.md`
4. Compares with previous run (if exists)

**Usage:**
```bash
python generate_test_coverage_summary.py
```

**Output:**
- Prints test summary to terminal
- Writes `TEST_COVERAGE_SUMMARY.md`
- Generates `htmlcov/` directory
- Creates `coverage.json`

**Exit codes:**
- `0`: All tests passed
- Non-zero: Some tests failed (check output for details)

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- Repository test documentation: `TEST_SUITE_DOCUMENTATION_README.md`
- Test analysis: `TEST_ANALYSIS_SUMMARY.md`

## Maintenance

### Updating Coverage Configuration

Edit `pyproject.toml` to modify:
- Excluded files/directories (`[tool.coverage.run] omit`)
- Excluded code lines (`[tool.coverage.report] exclude_lines`)
- Coverage precision and display options

### Adding New Test Markers

1. Add marker to `pyproject.toml`:
   ```toml
   [tool.pytest.ini_options]
   markers = [
       "your_marker: description here",
   ]
   ```

2. Use in tests:
   ```python
   import pytest
   
   @pytest.mark.your_marker
   def test_something():
       pass
   ```

### Updating This Documentation

This file should be updated when:
- New convenience scripts are added
- Coverage goals change
- New test markers are introduced
- CI/CD integration changes

---

**Last Updated:** 2024  
**Maintainer:** Neural DSL Team
