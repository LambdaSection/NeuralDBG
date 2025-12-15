# Neural DSL - Test Commands Cheatsheet

Quick reference card for testing commands. Print or keep handy during development.

---

## 🚀 Quick Run

```bash
# Generate full coverage report (RECOMMENDED)
python generate_test_coverage_summary.py

# Or use convenience scripts
.\run_tests_with_coverage.bat          # Windows
./run_tests_with_coverage.sh           # Unix/Linux/macOS

# Or use Make
make test-cov-report
```

---

## 🧪 Basic Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=neural

# Run with HTML report
pytest tests/ -v --cov=neural --cov-report=html

# Stop on first failure
pytest tests/ -v -x

# Show local variables on failure
pytest tests/ -v --showlocals
```

---

## 🎯 Selective Testing

```bash
# Specific file
pytest tests/parser/test_parser.py -v

# Specific test function
pytest tests/parser/test_parser.py::test_parse_simple_model -v

# Specific module coverage
pytest tests/parser/ -v --cov=neural.parser

# Only fast tests
pytest tests/ -v -m "not slow"

# Only unit tests
pytest tests/ -v -m unit

# Without TensorFlow
pytest tests/ -v -m "not requires_tensorflow"
```

---

## 🔄 Rerun Helpers

```bash
# Rerun last failed tests
pytest tests/ -v --lf

# Run failed first, then others
pytest tests/ -v --ff

# Show test collection without running
pytest --collect-only
```

---

## 📊 View Results

```bash
# View markdown summary
cat TEST_COVERAGE_SUMMARY.md

# Open HTML report
start htmlcov/index.html          # Windows
open htmlcov/index.html           # macOS  
xdg-open htmlcov/index.html       # Linux

# View JSON data
cat coverage.json | python -m json.tool
```

---

## 🛠️ Make Commands

```bash
make test              # Basic test run
make test-cov          # Tests with coverage
make test-cov-report   # Full coverage report
make lint              # Run linters
make format            # Format code
```

---

## 🏷️ Test Markers

| Marker | Usage |
|--------|-------|
| `slow` | `-m "not slow"` |
| `integration` | `-m integration` |
| `unit` | `-m unit` |
| `requires_gpu` | `-m "not requires_gpu"` |
| `requires_torch` | `-m "not requires_torch"` |
| `requires_tensorflow` | `-m "not requires_tensorflow"` |
| `requires_onnx` | `-m "not requires_onnx"` |

---

## 🐛 Troubleshooting

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install package in dev mode
pip install -e .

# Install with all features
pip install -e ".[full]"

# Install pytest-cov
pip install pytest-cov

# Check pytest version
pytest --version

# Verify test discovery
pytest --collect-only tests/
```

---

## 🔧 Common Workflows

### Before Commit
```bash
pytest tests/ -v              # Run tests
ruff check .                  # Lint
ruff check --fix .            # Auto-fix
mypy neural/ --ignore-missing-imports  # Type check
```

### After Changes
```bash
pytest tests/ -v --cov=neural --cov-report=term
# Check coverage percentage
```

### Generate Report
```bash
python generate_test_coverage_summary.py
cat TEST_COVERAGE_SUMMARY.md
```

### Debug Failing Test
```bash
pytest tests/path/to/test.py::test_name -vv --tb=long --showlocals
```

---

## 📈 Coverage Goals

| Module | Target |
|--------|--------|
| Overall | 90% |
| parser/ | 95% |
| code_generation/ | 95% |
| shape_propagation/ | 95% |

---

## 📚 Documentation

- Quick Reference: `TESTING_QUICK_REFERENCE.md`
- Complete Guide: `TEST_COVERAGE_README.md`
- System Overview: `TESTING_SYSTEM_OVERVIEW.md`
- Documentation Index: `TESTING_INDEX.md`

---

## 💡 Tips

- Mark slow tests: `@pytest.mark.slow`
- Use fixtures for shared setup
- One assertion per test (when possible)
- Descriptive test names
- Check coverage for new code
- Run fast tests during development
- Run full suite before PR

---

**Print this page for quick reference during development!**
