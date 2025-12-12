# Building Neural DSL Documentation

This guide explains how to build the Neural DSL documentation using Sphinx.

## Prerequisites

Install documentation dependencies:

```bash
# Install with docs extras
pip install -e ".[docs]"

# Or install manually
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

## Building HTML Documentation

### On Unix/Linux/Mac

```bash
cd docs
make html
```

### On Windows

```bash
cd docs
.\make.bat html
```

## Viewing the Documentation

After building, open the documentation in your browser:

```bash
# On Unix/Linux/Mac
open _build/html/index.html

# On Windows
start _build/html/index.html
```

## Building Other Formats

Sphinx supports multiple output formats:

```bash
# PDF (requires LaTeX)
make latexpdf

# ePub
make epub

# Man pages
make man

# Plain text
make text
```

## Clean Build

To remove all built documentation:

```bash
# Unix/Linux/Mac
make clean

# Windows
.\make.bat clean
```

## Live Preview with Auto-Reload

For development, you can use sphinx-autobuild:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build/html
```

Then open http://localhost:8000 in your browser. The documentation will
automatically rebuild when you make changes.

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index
- `api/` - API reference documentation
  - Auto-generated from docstrings using sphinx.ext.autodoc
  - Follows NumPy/Google docstring style
- `_static/` - Static files (CSS, images)
- `_templates/` - Custom Sphinx templates
- `_build/` - Generated documentation (gitignored)

## Docstring Style

Neural DSL uses NumPy-style docstrings. Example:

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """
    Brief description of the function.
    
    More detailed description if needed.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str, optional
        Description of param2, by default "default"
        
    Returns
    -------
    bool
        Description of return value
        
    Raises
    ------
    ValueError
        Description of when this is raised
        
    Examples
    --------
    >>> example_function(42, "test")
    True
    
    Notes
    -----
    Additional notes or references.
    """
    return True
```

## Updating API Documentation

When you add new modules or classes:

1. Add docstrings following NumPy style
2. Update the relevant `.rst` file in `docs/api/`
3. Rebuild the documentation

## Troubleshooting

### Import Errors

If Sphinx can't import modules, ensure Neural DSL is installed:

```bash
pip install -e .
```

### Missing Dependencies

If you see warnings about missing extensions:

```bash
pip install -e ".[docs]"
```

### Theme Not Found

If the RTD theme is not found:

```bash
pip install sphinx-rtd-theme
```
