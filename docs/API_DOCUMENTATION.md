# Neural DSL API Documentation Guide

## Overview

Neural DSL now includes comprehensive API documentation using Sphinx with autodoc extension. The documentation follows NumPy/Google docstring style conventions and includes type hints for all public APIs.

## What Has Been Added

### 1. Sphinx Configuration (`docs/conf.py`)

- Complete Sphinx setup with autodoc, napoleon, and type hints extensions
- RTD (Read the Docs) theme configured
- Intersphinx mapping for Python, NumPy, PyTorch, and TensorFlow docs
- Auto-summary generation enabled

### 2. API Reference Structure (`docs/api/`)

Complete API documentation organized by module:
- `index.rst` - Main API index
- `neural.rst` - Core package documentation
- `parser.rst` - DSL parser and validation
- `code_generation.rst` - Multi-backend code generation
- `shape_propagation.rst` - Shape inference and validation
- `cli.rst` - Command-line interface
- `dashboard.rst` - Real-time training dashboard
- `hpo.rst` - Hyperparameter optimization
- `cloud.rst` - Cloud deployment (Kaggle, Colab, SageMaker)
- `utils.rst` - Utility functions
- `visualization.rst` - Network visualization

### 3. Build Infrastructure

- `docs/Makefile` - Unix/Linux/Mac build scripts
- `docs/make.bat` - Windows build scripts
- `docs/requirements.txt` - Documentation dependencies
- `docs/BUILD_DOCS.md` - Comprehensive build instructions

### 4. Enhanced Docstrings

All public functions and classes now include comprehensive docstrings:

#### Package Level (`neural/__init__.py`)
- Module overview with features list
- Detailed module descriptions
- Usage examples
- Enhanced `check_dependencies()` function documentation

#### Code Generation (`neural/code_generation/`)
- Module-level documentation
- `generate_code()` with full parameter and return documentation
- Helper functions with type hints and examples
- Error documentation

#### Shape Propagation (`neural/shape_propagation/`)
- `ShapePropagator` class with comprehensive docstring
- `propagate()` method with parameters, returns, and examples
- `PerformanceMonitor` class documentation
- Helper methods documented

#### HPO (`neural/hpo/`)
- Module overview with features
- `create_dynamic_model()` function
- `resolve_hpo_params()` function
- Data loading utilities

#### Cloud (`neural/cloud/`)
- `CloudExecutor` class with comprehensive documentation
- Environment detection methods
- Model compilation and training methods
- Examples for each cloud platform

#### Utils (`neural/utils/`)
- Seeding functions for reproducibility
- Type-annotated functions
- Usage examples

### 5. Type Hints

All public APIs now include proper type hints:
- Function parameters with types
- Return types specified
- Optional parameters clearly marked
- Generic types (Dict, List, Tuple, Optional) properly used

### 6. Documentation Standards

All docstrings follow NumPy style with:

```python
"""
Brief one-line description.

More detailed description if needed.

Parameters
----------
param1 : type
    Description
param2 : type, optional
    Description, by default value

Returns
-------
type
    Description

Raises
------
ExceptionType
    When and why

Examples
--------
>>> example_code()
result

Notes
-----
Additional information
"""
```

## Building the Documentation

### Installation

```bash
# Install with documentation dependencies
pip install -e ".[docs]"

# Or install manually
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

### Build Commands

```bash
# Unix/Linux/Mac
cd docs
make html

# Windows
cd docs
.\make.bat html
```

### View Documentation

```bash
# Open in browser
open docs/_build/html/index.html  # Mac
xdg-open docs/_build/html/index.html  # Linux
start docs/_build/html/index.html  # Windows
```

### Live Preview

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build/html
# Opens at http://localhost:8000
```

## Documentation Coverage

### Fully Documented Modules

✅ `neural` - Main package with dependency checking
✅ `neural.code_generation` - Code generation for all backends
✅ `neural.shape_propagation` - Shape inference and validation
✅ `neural.hpo` - Hyperparameter optimization
✅ `neural.cloud` - Cloud execution and deployment
✅ `neural.utils` - Utility functions and seeding
✅ `neural.parser.validation` - Parameter validation (already documented)
✅ `neural.parser.error_handling` - Error handling (already documented)

### Modules with Partial Documentation

These modules already had good docstrings but may need minor enhancements:
- `neural.parser.parser` - Main parser implementation
- `neural.cli.cli` - CLI commands
- `neural.dashboard` - Training dashboard
- `neural.visualization` - Network visualization

## Integration with setup.py

Added `docs` extra to `setup.py`:

```python
extras_require={
    "docs": [
        "sphinx>=5.0",
        "sphinx-rtd-theme>=1.0",
        "sphinx-autodoc-typehints>=1.19",
        "myst-parser>=0.18"
    ]
}
```

## Best Practices for Contributors

When adding new code:

1. **Always add docstrings** to public functions/classes
2. **Use NumPy style** for consistency
3. **Include type hints** on all parameters and returns
4. **Provide examples** in docstrings when helpful
5. **Document exceptions** that can be raised
6. **Update .rst files** in `docs/api/` for new modules
7. **Test documentation** builds locally before committing

## Quality Checks

Before committing:

```bash
# Check if docs build without errors
cd docs
make html

# Check for missing docstrings
python -c "import neural; help(neural)"

# Verify type hints
mypy neural/ --ignore-missing-imports
```

## Future Enhancements

Consider adding:
- Tutorials in `docs/tutorials/`
- API usage guides in `docs/guides/`
- Architecture diagrams
- Interactive examples
- Contribution guidelines in docs
- API changelog tracking
- Doctests for examples

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Google Docstring Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Type Hints PEP 484](https://www.python.org/dev/peps/pep-0484/)
- [Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
