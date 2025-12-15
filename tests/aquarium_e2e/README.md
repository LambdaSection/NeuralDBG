# Aquarium IDE End-to-End Tests

Comprehensive E2E tests for the Aquarium IDE using Playwright, verifying the complete user journey from welcome screen through template selection, DSL editing, compilation, debugging, and export.

## Overview

These tests cover the full workflow:

1. **Welcome Screen & Templates** - Loading and selecting example templates
2. **DSL Editor** - Writing and editing Neural DSL code
3. **Parsing & Validation** - Syntax checking and model validation
4. **Compilation** - Generating code for TensorFlow, PyTorch, and ONNX
5. **Debugging** - Real-time execution monitoring
6. **Export** - Exporting compiled models to files
7. **IDE Integration** - Opening exported models in external IDEs

## Test Structure

```
tests/aquarium_e2e/
├── __init__.py                    # Package initialization
├── conftest.py                    # Pytest fixtures and configuration
├── page_objects.py                # Page Object Models for UI interactions
├── utils.py                       # Utility functions for tests
├── test_dsl_editor.py            # DSL editor functionality tests
├── test_compilation.py           # Model compilation tests
├── test_export.py                # Export and IDE integration tests
├── test_complete_workflow.py     # Full end-to-end workflow tests
├── test_navigation.py            # Tab navigation and UI flow tests
├── test_ui_elements.py           # UI component tests
└── README.md                     # This file
```

## Setup

### Prerequisites

1. Install Playwright:
```bash
pip install playwright
playwright install chromium
```

2. Install test dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Install Aquarium IDE dependencies:
```bash
pip install -e ".[dashboard]"
```

### Environment Variables

- `HEADLESS`: Run browser in headless mode (default: `true`)
  ```bash
  export HEADLESS=false  # Run with visible browser
  ```

- `SLOW_MO`: Slow down operations by N milliseconds (default: `0`)
  ```bash
  export SLOW_MO=100  # Useful for debugging
  ```

## Running Tests

### Run all E2E tests:
```bash
pytest tests/aquarium_e2e/ -v
```

### Run specific test file:
```bash
pytest tests/aquarium_e2e/test_complete_workflow.py -v
```

### Run specific test:
```bash
pytest tests/aquarium_e2e/test_dsl_editor.py::TestDSLEditor::test_parse_valid_dsl -v
```

### Run with visible browser:
```bash
HEADLESS=false pytest tests/aquarium_e2e/ -v
```

### Run with slow motion (helpful for debugging):
```bash
HEADLESS=false SLOW_MO=500 pytest tests/aquarium_e2e/test_complete_workflow.py -v
```

### Run tests in parallel:
```bash
pytest tests/aquarium_e2e/ -v -n auto
```

### Run with screenshots on failure:
```bash
pytest tests/aquarium_e2e/ -v --screenshot=on
```

## Test Categories

### 1. DSL Editor Tests (`test_dsl_editor.py`)
- Editor loading and visibility
- Loading example code
- Parsing valid DSL
- Handling invalid DSL
- Editing DSL content
- Model information display

**Example:**
```bash
pytest tests/aquarium_e2e/test_dsl_editor.py -v
```

### 2. Compilation Tests (`test_compilation.py`)
- Compilation to TensorFlow
- Compilation to PyTorch
- Compilation to ONNX
- Console output verification
- Dataset selection
- Training configuration

**Example:**
```bash
pytest tests/aquarium_e2e/test_compilation.py::TestCompilation::test_compile_tensorflow -v
```

### 3. Export Tests (`test_export.py`)
- Export modal interactions
- Custom filename export
- Custom location export
- Multiple backend exports
- Success notifications

**Example:**
```bash
pytest tests/aquarium_e2e/test_export.py -v
```

### 4. Complete Workflow Tests (`test_complete_workflow.py`)
- Full DSL → Parse → Compile → Export workflow
- Multiple backend compilations
- Different dataset selections
- Tab navigation during workflow
- Error recovery

**Example:**
```bash
pytest tests/aquarium_e2e/test_complete_workflow.py::TestCompleteWorkflow::test_simple_workflow_tensorflow -v
```

### 5. Navigation Tests (`test_navigation.py`)
- Tab switching
- Active tab detection
- Content persistence across tabs

**Example:**
```bash
pytest tests/aquarium_e2e/test_navigation.py -v
```

### 6. UI Element Tests (`test_ui_elements.py`)
- Page title and header
- Button presence and states
- Form inputs
- Styling verification

**Example:**
```bash
pytest tests/aquarium_e2e/test_ui_elements.py -v
```

## Page Object Model

The tests use Page Object Model (POM) pattern for better maintainability:

- `AquariumIDEPage`: Base page class
- `DSLEditorPage`: DSL editor interactions
- `RunnerPanelPage`: Runner panel interactions
- `ExportModalPage`: Export modal interactions
- `NavigationPage`: Tab navigation
- `AquariumWorkflow`: High-level workflow orchestration

**Example usage:**
```python
from tests.aquarium_e2e.page_objects import AquariumWorkflow

def test_my_workflow(page):
    workflow = AquariumWorkflow(page)
    workflow.complete_basic_workflow(dsl_content, backend="tensorflow")
    workflow.export_model_script("my_model.py")
```

## Fixtures

### Available Fixtures (from `conftest.py`)

- `aquarium_server`: Starts Aquarium IDE server for testing
- `browser`: Playwright browser instance
- `context`: Browser context for test isolation
- `page`: Page instance with Aquarium loaded
- `screenshot_dir`: Directory for test screenshots
- `take_screenshot`: Helper function to take screenshots

**Example usage:**
```python
def test_example(page, take_screenshot):
    take_screenshot("initial_state")
    # ... perform actions
    take_screenshot("final_state")
```

## Debugging Tests

### 1. Run with visible browser and slow motion:
```bash
HEADLESS=false SLOW_MO=1000 pytest tests/aquarium_e2e/test_dsl_editor.py::TestDSLEditor::test_parse_valid_dsl -v -s
```

### 2. Use Playwright Inspector:
```bash
PWDEBUG=1 pytest tests/aquarium_e2e/test_complete_workflow.py::TestCompleteWorkflow::test_simple_workflow_tensorflow
```

### 3. Check screenshots:
Screenshots are saved to `tests/aquarium_e2e/screenshots/` directory on test failures.

### 4. Enable verbose logging:
```bash
pytest tests/aquarium_e2e/ -v -s --log-cli-level=DEBUG
```

## Common Issues

### Issue: Server fails to start
**Solution:** Check if port 8052 is already in use:
```bash
netstat -ano | findstr :8052  # Windows
lsof -i :8052                 # Linux/Mac
```

### Issue: Tests are too slow
**Solution:** Reduce timeout values or run specific tests instead of the full suite.

### Issue: Flaky tests
**Solution:** Increase wait times in `conftest.py` or use explicit waits in tests.

### Issue: Browser not found
**Solution:** Install Playwright browsers:
```bash
playwright install chromium
```

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

### GitHub Actions Example:
```yaml
name: E2E Tests
on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dashboard]"
          pip install -r requirements-dev.txt
          playwright install chromium
      - name: Run E2E tests
        run: pytest tests/aquarium_e2e/ -v
      - name: Upload screenshots
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-screenshots
          path: tests/aquarium_e2e/screenshots/
```

## Best Practices

1. **Use Page Objects**: Encapsulate page interactions in page object classes
2. **Wait Appropriately**: Use explicit waits instead of sleep
3. **Take Screenshots**: Use `take_screenshot` fixture for debugging
4. **Isolate Tests**: Each test should be independent
5. **Clean Up**: Use fixtures to ensure proper cleanup
6. **Descriptive Names**: Use clear, descriptive test names
7. **Test Data**: Use parameterized tests for testing multiple scenarios

## Extending Tests

### Adding New Test Cases:

1. Create a new test file or add to existing ones
2. Import necessary page objects
3. Write test using page objects and fixtures
4. Add appropriate assertions
5. Document the test purpose

**Example:**
```python
from tests.aquarium_e2e.page_objects import DSLEditorPage

class TestNewFeature:
    def test_new_functionality(self, page, take_screenshot):
        editor = DSLEditorPage(page)
        
        # Test implementation
        editor.set_dsl_content(sample_dsl)
        take_screenshot("feature_test")
        
        # Assertions
        assert editor.is_parse_successful()
```

## Performance Benchmarks

Expected test execution times:

- Individual test: 5-15 seconds
- Test file: 1-5 minutes
- Full suite: 10-20 minutes

Slow tests are marked with `@pytest.mark.slow` and can be skipped:
```bash
pytest tests/aquarium_e2e/ -v -m "not slow"
```

## Support

For issues or questions:
1. Check this README
2. Review test code and page objects
3. Check screenshots in failures
4. Review Playwright documentation: https://playwright.dev/python/

## License

Same as main Neural DSL project.
