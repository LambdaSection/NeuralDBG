# Aquarium E2E Tests - Troubleshooting Guide

Common issues and solutions for Aquarium IDE end-to-end tests.

## Installation Issues

### Issue: Playwright not found
```
ModuleNotFoundError: No module named 'playwright'
```

**Solution:**
```bash
pip install playwright
playwright install chromium
```

### Issue: Browser binary not found
```
Error: Executable doesn't exist at ...
```

**Solution:**
```bash
playwright install chromium
# On Linux, also install system dependencies:
playwright install-deps chromium
```

### Issue: Dash not installed
```
ModuleNotFoundError: No module named 'dash'
```

**Solution:**
```bash
pip install -e ".[dashboard]"
# Or install full dependencies:
pip install -e ".[full]"
```

## Server Issues

### Issue: Port 8052 already in use
```
RuntimeError: Aquarium server failed to start
```

**Solution:**

**Windows:**
```powershell
netstat -ano | findstr :8052
taskkill /PID <PID> /F
```

**Linux/Mac:**
```bash
lsof -ti:8052 | xargs kill -9
```

### Issue: Server won't start in CI
```
TimeoutError: Server startup timeout
```

**Solution:**
1. Increase `SERVER_STARTUP_TIMEOUT` in conftest.py
2. Check if all dependencies are installed
3. Verify server can start manually:
```bash
python -m neural.aquarium.aquarium --port 8052
```

### Issue: Health check fails
```
requests.exceptions.ConnectionError
```

**Solution:**
1. Verify server is running
2. Check firewall settings
3. Ensure port is not blocked
4. Try different port in conftest.py

## Test Execution Issues

### Issue: Tests fail with "Element not found"
```
TimeoutError: Timeout 10000ms exceeded
```

**Solution:**
1. Increase timeout values in test
2. Run with visible browser to debug:
```bash
python tests/aquarium_e2e/run_tests.py --visible
```
3. Check if UI elements changed
4. Update selectors in `page_objects.py`

### Issue: Tests are very slow
```
Tests take 30+ minutes to complete
```

**Solution:**
1. Run fast tests only:
```bash
python tests/aquarium_e2e/run_tests.py --fast
```
2. Run in parallel:
```bash
python tests/aquarium_e2e/run_tests.py --parallel
```
3. Run specific test files:
```bash
pytest tests/aquarium_e2e/test_ui_elements.py -v
```

### Issue: Random test failures (flaky tests)
```
Tests pass sometimes, fail other times
```

**Solution:**
1. Increase wait times
2. Add explicit waits:
```python
page.wait_for_selector("#element", state="visible")
```
3. Use `wait_for_load_state`:
```python
page.wait_for_load_state("networkidle")
```
4. Run failed test multiple times to verify

### Issue: Screenshot directory not found
```
FileNotFoundError: No such directory
```

**Solution:**
```bash
mkdir -p tests/aquarium_e2e/screenshots
```

## Browser Issues

### Issue: Browser crashes during tests
```
Error: Browser closed unexpectedly
```

**Solution:**
1. Update Playwright:
```bash
pip install --upgrade playwright
playwright install chromium
```
2. Increase system resources (RAM/CPU)
3. Reduce parallel test count
4. Check browser logs

### Issue: Headless mode behaves differently
```
Tests pass in visible mode, fail in headless
```

**Solution:**
1. Some UI elements may render differently
2. Add delays for animations:
```python
page.wait_for_timeout(500)
```
3. Ensure fonts are loaded:
```python
page.wait_for_load_state("networkidle")
```

## CI/CD Issues

### Issue: GitHub Actions workflow fails
```
Error: Process completed with exit code 1
```

**Solution:**
1. Check workflow logs for specific error
2. Verify all dependencies in workflow
3. Ensure Playwright browsers installed:
```yaml
- name: Install Playwright browsers
  run: playwright install chromium
```
4. Add system dependencies on Linux:
```yaml
- name: Install deps
  run: playwright install-deps chromium
```

### Issue: Artifacts not uploaded
```
Warning: No files were found with the provided path
```

**Solution:**
1. Ensure screenshots directory exists
2. Check artifact path in workflow:
```yaml
path: tests/aquarium_e2e/screenshots/
```
3. Verify tests actually create screenshots

### Issue: Matrix builds fail on Windows
```
Tests pass on Ubuntu, fail on Windows
```

**Solution:**
1. Check path separators (use `Path` from pathlib)
2. Verify command syntax (PowerShell vs bash)
3. Check file permissions
4. Test locally on Windows

## Compilation Issues

### Issue: Compilation timeout
```
TimeoutError: Compilation did not complete
```

**Solution:**
1. Increase compilation timeout:
```python
runner.wait_for_compilation(timeout=60000)
```
2. Check if backend is installed
3. Verify model complexity
4. Check system resources

### Issue: Backend not available
```
Error: Backend 'pytorch' not found
```

**Solution:**
```bash
pip install torch
# Or install full backends:
pip install -e ".[backends]"
```

## Export Issues

### Issue: Export directory not created
```
FileNotFoundError: Export directory does not exist
```

**Solution:**
1. Ensure directory creation in test:
```python
export_dir.mkdir(parents=True, exist_ok=True)
```
2. Check write permissions
3. Use absolute paths

### Issue: Exported file not found
```
AssertionError: Exported file not found
```

**Solution:**
1. Add delay after export:
```python
page.wait_for_timeout(1000)
```
2. Verify export path is correct
3. Check export notification for errors

## Performance Issues

### Issue: Page load is slow
```
Page takes 30+ seconds to load
```

**Solution:**
1. Check network conditions
2. Verify server resources
3. Use `wait_for_load_state` appropriately
4. Consider mocking heavy resources

### Issue: Memory leaks during tests
```
Memory usage keeps increasing
```

**Solution:**
1. Close contexts properly:
```python
context.close()
```
2. Clear browser cache between tests
3. Restart browser periodically
4. Reduce parallel test count

## Debug Techniques

### 1. Visual Debugging
```bash
HEADLESS=false SLOW_MO=1000 pytest tests/aquarium_e2e/test_dsl_editor.py -v
```

### 2. Playwright Inspector
```bash
PWDEBUG=1 pytest tests/aquarium_e2e/test_complete_workflow.py::test_simple_workflow_tensorflow
```

### 3. Take Screenshots
```python
def test_something(page, take_screenshot):
    take_screenshot("before_action")
    # ... perform action
    take_screenshot("after_action")
```

### 4. Print Console Logs
```python
page.on("console", lambda msg: print(f"[CONSOLE] {msg.text}"))
```

### 5. Enable Verbose Logging
```bash
pytest tests/aquarium_e2e/ -v -s --log-cli-level=DEBUG
```

### 6. Use Debugger
```python
import pdb; pdb.set_trace()
# Or with pytest:
pytest --pdb
```

### 7. Check Element State
```python
element = page.locator("#my-element")
print(f"Visible: {element.is_visible()}")
print(f"Enabled: {element.is_enabled()}")
print(f"Text: {element.inner_text()}")
```

## Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `TimeoutError` | Element not appearing | Increase timeout or check selector |
| `ConnectionError` | Server not running | Start server or check health endpoint |
| `ModuleNotFoundError` | Missing dependency | Install required package |
| `FileNotFoundError` | Missing file/directory | Create directory or check path |
| `AssertionError` | Test expectation failed | Debug with screenshots |
| `RuntimeError` | Server startup failed | Check port availability |

## Getting Help

1. **Check Documentation**: Review README.md and QUICKSTART.md
2. **Search Issues**: Look for similar problems in test output
3. **Enable Debug Mode**: Run with `--debug` flag
4. **Check Logs**: Review console output and screenshots
5. **Simplify Test**: Isolate the failing part
6. **Ask for Help**: Provide error message, screenshots, and steps to reproduce

## Reporting Issues

When reporting issues, include:

1. Error message and stack trace
2. Test command used
3. Python version: `python --version`
4. Playwright version: `playwright --version`
5. Operating system
6. Screenshots from failures
7. Minimal reproducible example

Example:
```
Environment:
- Python 3.11
- Playwright 1.40.0
- Ubuntu 22.04
- Test: test_complete_workflow.py::test_simple_workflow_tensorflow

Error:
TimeoutError: Timeout 10000ms exceeded waiting for selector "#runner-status-badge"

Command:
pytest tests/aquarium_e2e/test_complete_workflow.py::test_simple_workflow_tensorflow -v

Screenshots: attached
```

## Prevention Tips

1. **Use Explicit Waits**: Always wait for elements to be ready
2. **Keep Selectors Stable**: Use IDs or data-testid attributes
3. **Test Locally First**: Before pushing to CI
4. **Monitor Timeouts**: Adjust based on system performance
5. **Clean Up Resources**: Close contexts and browsers
6. **Version Lock**: Pin Playwright version in requirements
7. **Update Regularly**: Keep dependencies current
8. **Document Changes**: Note UI changes that affect tests

## Quick Fixes Cheat Sheet

```bash
# Reinstall everything
pip uninstall playwright -y
pip install playwright
playwright install chromium

# Clear caches
rm -rf .pytest_cache/
rm -rf tests/aquarium_e2e/screenshots/*
rm -rf tests/aquarium_e2e/.pytest_cache/

# Check server manually
python -m neural.aquarium.aquarium --port 8052

# Run minimal test
pytest tests/aquarium_e2e/test_ui_elements.py::TestUIElements::test_page_title -v

# Run with all debug options
HEADLESS=false SLOW_MO=1000 PWDEBUG=1 pytest tests/aquarium_e2e/test_dsl_editor.py -v -s

# Check port usage (Windows)
netstat -ano | findstr :8052

# Check port usage (Linux/Mac)
lsof -ti:8052
```

---

**Still having issues?** Check the complete README.md or create an issue with full details.
