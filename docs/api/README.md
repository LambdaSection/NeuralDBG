# Neural DSL API Documentation

This directory contains the API reference documentation for Neural DSL, 
auto-generated using Sphinx with autodoc.

## Building the Documentation

To build the HTML documentation:

```bash
cd docs
pip install -r requirements.txt
make html
```

On Windows:
```bash
cd docs
pip install -r requirements.txt
.\make.bat html
```

The generated documentation will be in `docs/_build/html/`.

## Documentation Structure

- `index.rst` - Main API reference index
- `neural.rst` - Main package documentation
- `parser.rst` - Parser module API
- `code_generation.rst` - Code generation API
- `shape_propagation.rst` - Shape propagation API
- `cli.rst` - CLI module API
- `dashboard.rst` - Dashboard API
- `hpo.rst` - HPO module API
- `cloud.rst` - Cloud integration API
- `utils.rst` - Utilities API
- `visualization.rst` - Visualization API

## Viewing the Documentation

After building, open `docs/_build/html/index.html` in your browser.

For development with auto-reload:
```bash
cd docs
sphinx-autobuild . _build/html
```
