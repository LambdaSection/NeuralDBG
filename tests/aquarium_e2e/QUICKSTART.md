# Aquarium E2E Tests - Quick Start Guide

Get started with Aquarium IDE end-to-end tests in 5 minutes.

## Prerequisites

- Python 3.9+
- Neural DSL repository cloned
- Basic familiarity with pytest

## 1. Install Dependencies

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Install Playwright browsers
playwright install chromium

# Install Aquarium dependencies
pip install -e ".[dashboard]"
```

## 2. Verify Installation

```bash
# Check Playwright installation
playwright --version

# Check pytest installation
pytest --version

# Verify imports
python -c "import playwright; import dash; print('✓ All dependencies installed')"
```

## 3. Run Your First Test

```bash
# Run a single simple test
pytest tests/aquarium_e2e/test_ui_elements.py::TestUIElements::test_page_title -v

# Expected output:
# tests/aquarium_e2e/test_ui_elements.py::TestUIElements::test_page_title PASSED
```

## 4. Run All Tests

```bash
# Using the run script (recommended)
python tests/aquarium_e2e/run_tests.py

# Or directly with pytest
pytest tests/aquarium_e2e/ -v
```

## 5. Run Tests Visually (Debug Mode)

```bash
# Run with visible browser
python tests/aquarium_e2e/run_tests.py --visible

# Run with visible browser and slow motion
python tests/aquarium_e2e/run_tests.py --visible --slow-mo 500

# Run with Playwright Inspector (best for debugging)
python tests/aquarium_e2e/run_tests.py --debug
```

## 6. Run Specific Test Categories

```bash
# Run only DSL editor tests
python tests/aquarium_e2e/run_tests.py --file test_dsl_editor.py

# Run only compilation tests
python tests/aquarium_e2e/run_tests.py --file test_compilation.py

# Run complete workflow tests
python tests/aquarium_e2e/run_tests.py --file test_complete_workflow.py

# Skip slow tests
python tests/aquarium_e2e/run_tests.py --fast

# Run tests in parallel (faster)
python tests/aquarium_e2e/run_tests.py --parallel
```

## 7. Understanding Test Results

### ✅ Success
```
tests/aquarium_e2e/test_dsl_editor.py::TestDSLEditor::test_parse_valid_dsl PASSED
```

### ❌ Failure
```
tests/aquarium_e2e/test_dsl_editor.py::TestDSLEditor::test_parse_valid_dsl FAILED
```
- Check `tests/aquarium_e2e/screenshots/` for failure screenshots
- Review error message in console output

### ⚠️ Skipped
```
tests/aquarium_e2e/test_performance.py::TestPerformance::test_compilation_performance SKIPPED
```
- Test was skipped (usually slow tests when using `--fast`)

## 8. Common Commands

```bash
# Quick smoke test
pytest tests/aquarium_e2e/test_ui_elements.py -v

# Full test suite
python tests/aquarium_e2e/run_tests.py

# Debug a failing test
python tests/aquarium_e2e/run_tests.py --debug --file test_dsl_editor.py

# Run and generate coverage report
pytest tests/aquarium_e2e/ --cov=neural.aquarium --cov-report=html

# Run specific test
pytest tests/aquarium_e2e/test_complete_workflow.py::TestCompleteWorkflow::test_simple_workflow_tensorflow -v
```

## 9. Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8052 (Windows)
netstat -ano | findstr :8052
taskkill /PID <PID> /F

# Kill process on port 8052 (Linux/Mac)
lsof -ti:8052 | xargs kill -9
```

### Browser Not Found
```bash
# Reinstall browsers
playwright install chromium
```

### Server Won't Start
```bash
# Check if dependencies are installed
pip install -e ".[dashboard]"

# Manually test server
python -m neural.aquarium.aquarium --port 8052
```

### Tests Are Slow
```bash
# Run in parallel
python tests/aquarium_e2e/run_tests.py --parallel

# Skip slow tests
python tests/aquarium_e2e/run_tests.py --fast
```

## 10. Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [page_objects.py](page_objects.py) to understand test structure
- Check [conftest.py](conftest.py) for available fixtures
- Review existing tests for examples
- Write your own tests using the Page Object Model

## Test File Overview

| File | Description | Time |
|------|-------------|------|
| `test_ui_elements.py` | UI component tests | ~30s |
| `test_dsl_editor.py` | DSL editor functionality | ~1m |
| `test_navigation.py` | Tab navigation | ~30s |
| `test_compilation.py` | Model compilation | ~2m |
| `test_export.py` | Export functionality | ~1m |
| `test_complete_workflow.py` | End-to-end workflows | ~3m |
| `test_performance.py` | Performance tests (slow) | ~5m |

Total suite time: ~10-15 minutes

## Getting Help

1. Check this guide
2. Review [README.md](README.md) for detailed documentation
3. Look at existing tests for examples
4. Check Playwright documentation: https://playwright.dev/python/
5. Review pytest documentation: https://docs.pytest.org/

## Quick Reference Card

```bash
# Essential Commands
pytest tests/aquarium_e2e/                                # Run all tests
python tests/aquarium_e2e/run_tests.py --visible         # Visual mode
python tests/aquarium_e2e/run_tests.py --debug           # Debug mode
python tests/aquarium_e2e/run_tests.py --fast            # Skip slow
python tests/aquarium_e2e/run_tests.py --parallel        # Parallel

# Windows
run_tests.bat
run_tests.bat --visible

# Linux/Mac
./run_tests.sh
./run_tests.sh --visible
```

---

**Ready to test!** 🚀

Start with: `python tests/aquarium_e2e/run_tests.py --visible --fast`
