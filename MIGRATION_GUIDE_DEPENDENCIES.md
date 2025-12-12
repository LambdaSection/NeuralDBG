# Dependency Management Migration Guide

This guide helps existing Neural DSL users transition to the new modular dependency structure.

## What Changed?

Neural DSL now uses a **modular dependency system** that separates core functionality from optional features. This allows you to install only what you need.

## Quick Summary

| What You Want | Old Command | New Command |
|--------------|-------------|-------------|
| Everything | `pip install -r requirements.txt` | `pip install neural-dsl[full]` |
| Core + PyTorch | N/A | `pip install neural-dsl[backends]` |
| Core only | N/A | `pip install neural-dsl` |
| Development | `pip install -e .` + manual tools | `pip install -r requirements-dev.txt` |

## For Regular Users

### Scenario 1: I was using `pip install neural-dsl`

**Nothing changes for you!** The basic installation still works. Now it's even lighter (only core dependencies).

If you need additional features:
```bash
# Add backends when you need them
pip install neural-dsl[backends]

# Or add specific features
pip install neural-dsl[visualization]
pip install neural-dsl[dashboard]
```

### Scenario 2: I was using `pip install -r requirements.txt`

**Before**:
```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -r requirements.txt  # Installed everything (~8GB)
```

**After (option 1 - same as before)**:
```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -e ".[full]"  # Installs everything (~8GB)
```

**After (option 2 - recommended)**:
```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -e .  # Core only (~20MB)

# Then add only what you need:
pip install -e ".[backends]"       # For PyTorch/TensorFlow
pip install -e ".[visualization]"  # For charts/diagrams
pip install -e ".[dashboard]"      # For NeuralDbg
```

### Scenario 3: I was using `pip install -e .`

**Before**: This installed all dependencies from `install_requires`

**After**: This now installs only core dependencies

**Migration**:
```bash
# If you need all features (like before):
pip install -e ".[full]"

# Or install specific features:
pip install -e ".[backends,visualization,dashboard]"
```

## For Contributors/Developers

### Scenario 4: I'm contributing to Neural DSL

**Before**:
```bash
git clone https://github.com/your-username/Neural.git
cd Neural
pip install -e .
pip install pytest ruff pylint mypy  # Manual tool installation
```

**After**:
```bash
git clone https://github.com/your-username/Neural.git
cd Neural
pip install -r requirements-dev.txt  # Installs everything needed
pre-commit install
```

This gives you:
- Core package in editable mode
- All development tools (pytest, ruff, pylint, mypy, pre-commit, pip-audit)
- Consistent environment with other contributors

### Scenario 5: I'm testing specific features

**Before**: Had to install everything

**After**: Install only what you're testing
```bash
# Testing PyTorch code generation
pip install -e ".[backends]"

# Testing HPO
pip install -e ".[hpo]"

# Testing dashboard
pip install -e ".[dashboard,visualization]"
```

## For CI/CD Pipelines

### Scenario 6: GitHub Actions workflows

**Before**:
```yaml
- name: Install dependencies
  run: pip install -r requirements.txt
```

**After (option 1 - full tests)**:
```yaml
- name: Install dependencies
  run: pip install -e ".[full]"
```

**After (option 2 - targeted tests)**:
```yaml
- name: Install dependencies
  run: |
    pip install -e .
    pip install -e ".[backends]"  # Only for backend tests
```

### Scenario 7: Docker images

**Before**:
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
```

**After (option 1 - same as before)**:
```dockerfile
RUN pip install neural-dsl[full]
```

**After (option 2 - optimized layers)**:
```dockerfile
# Core dependencies (cached layer)
RUN pip install neural-dsl

# Backends (separate layer for better caching)
RUN pip install neural-dsl[backends]
```

## Common Questions

### Q: Will my existing code break?

**A: No.** All functionality remains the same. Only the installation mechanism has changed.

### Q: Do I need to reinstall?

**A: No.** If your current installation works, you don't need to change anything. The new structure only affects new installations.

### Q: What if I already have TensorFlow/PyTorch installed?

**A: No problem.** The extras don't conflict with existing installations. pip will skip already-installed packages.

### Q: Can I still use `requirements.txt`?

**A: Yes.** But it now contains only core dependencies. For full installation, use `pip install neural-dsl[full]`.

### Q: How do I update to the latest version?

```bash
# Update with current feature set
pip install --upgrade neural-dsl[full]

# Or update with specific features
pip install --upgrade neural-dsl[backends,visualization]
```

### Q: I'm getting ImportError for torch/tensorflow

**A: Install the backends extra:**
```bash
pip install neural-dsl[backends]
```

### Q: I'm getting ImportError for dash/flask

**A: Install the dashboard extra:**
```bash
pip install neural-dsl[dashboard]
```

### Q: How do I know what features I have installed?

```bash
pip show neural-dsl
pip list | grep -E "(torch|tensorflow|dash|optuna)"
```

## Verification After Migration

After migrating, verify your setup:

```bash
# Check Neural DSL is installed
neural --version

# Test basic functionality
neural --help

# Test compilation (core feature)
neural compile examples/mnist.neural --backend tensorflow

# If you installed backends, test execution
python generated_code.py

# If you installed visualization, test that
neural visualize examples/mnist.neural

# If you installed dashboard, test that
neural debug examples/mnist.neural
```

## Rollback (if needed)

If you encounter issues and need to go back:

```bash
# Uninstall current installation
pip uninstall neural-dsl

# Reinstall with all features
pip install neural-dsl[full]
```

## Getting Help

If you encounter issues during migration:

1. **Check the dependency guide**: [DEPENDENCY_GUIDE.md](DEPENDENCY_GUIDE.md)
2. **Check installation reference**: [INSTALL.md](INSTALL.md)
3. **Open an issue**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
4. **Ask on Discord**: https://discord.gg/KFku4KvS

## Feature Groups Reference

Quick reference for what each feature group includes:

| Feature Group | Includes | Install Command |
|--------------|----------|-----------------|
| **core** | click, lark, numpy, pyyaml | `pip install neural-dsl` |
| **backends** | TensorFlow, PyTorch, ONNX | `pip install neural-dsl[backends]` |
| **hpo** | Optuna, scikit-learn | `pip install neural-dsl[hpo]` |
| **visualization** | matplotlib, graphviz, plotly | `pip install neural-dsl[visualization]` |
| **dashboard** | Dash, Flask, SocketIO | `pip install neural-dsl[dashboard]` |
| **cloud** | pygithub, selenium, tweepy | `pip install neural-dsl[cloud]` |
| **api** | FastAPI | `pip install neural-dsl[api]` |
| **utils** | psutil, pandas, scipy | `pip install neural-dsl[utils]` |
| **ml-extras** | HuggingFace, Transformers | `pip install neural-dsl[ml-extras]` |
| **dev** | pytest, ruff, mypy, pre-commit | `pip install neural-dsl[dev]` |
| **full** | All of the above | `pip install neural-dsl[full]` |

## Benefits of the New Structure

1. **Faster installation**: Install only what you need
2. **Smaller size**: Core installation is ~20MB vs ~8GB
3. **Clearer dependencies**: Know exactly what each feature requires
4. **Better for CI/CD**: Faster builds with targeted installations
5. **More flexible**: Add features incrementally as needed

## Examples

### Example 1: Data Scientist (PyTorch user)

**Before**:
```bash
pip install neural-dsl  # Got everything including TensorFlow (not needed)
```

**After**:
```bash
pip install neural-dsl  # Core only
pip install neural-dsl[backends]  # When ready to train
# Now has PyTorch, TensorFlow, ONNX
```

### Example 2: Student Learning DSL

**Before**:
```bash
pip install neural-dsl  # Had to download 8GB
```

**After**:
```bash
pip install neural-dsl  # Only 20MB, can start learning immediately
```

### Example 3: ML Engineer (Full Stack)

**Before**:
```bash
pip install -r requirements.txt
```

**After**:
```bash
pip install neural-dsl[full]  # Same result, cleaner command
```

### Example 4: Researcher (Using Colab)

**Before**:
```python
!pip install neural-dsl  # Took 5-10 minutes
```

**After**:
```python
!pip install neural-dsl[backends,visualization,cloud]  # Only what's needed
```

## Summary

The new dependency structure is **backward compatible** and provides **more flexibility**. You can:
- Continue using the old method (`pip install neural-dsl[full]`)
- Or adopt the new modular approach for faster, lighter installations

Choose what works best for your use case!
