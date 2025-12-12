# Neural DSL Documentation

This directory contains the complete documentation for Neural DSL, including user guides, API reference, and development documentation.

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation index
├── requirements.txt       # Documentation build dependencies
├── Makefile              # Unix/Linux/Mac build scripts
├── make.bat              # Windows build scripts
├── BUILD_DOCS.md         # Build instructions
├── API_DOCUMENTATION.md  # API documentation guide
├── DOCSTRING_GUIDE.md    # Docstring style guide
├── api/                  # API reference documentation
│   ├── index.rst
│   ├── neural.rst
│   ├── parser.rst
│   ├── code_generation.rst
│   ├── shape_propagation.rst
│   ├── cli.rst
│   ├── dashboard.rst
│   ├── hpo.rst
│   ├── cloud.rst
│   ├── utils.rst
│   ├── visualization.rst
│   └── README.md
├── _static/              # Static files (CSS, images)
├── _templates/           # Custom Sphinx templates
└── _build/               # Generated documentation (gitignored)
```

## Quick Start

### Install Dependencies

```bash
pip install -e ".[docs]"
```

### Build HTML Documentation

```bash
cd docs
make html  # Unix/Linux/Mac
.\make.bat html  # Windows
```

### View Documentation

```bash
open _build/html/index.html  # Mac
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

## Documentation Components

### User Documentation
- **installation.md** - Installation instructions
- **quickstart.ipynb** - Quick start guide
- **dsl.md** - DSL syntax reference
- **cli.md** - CLI command reference

### API Reference (`api/`)
Auto-generated from docstrings using Sphinx autodoc:
- Complete API reference for all modules
- Class and function documentation
- Type hints and examples
- Cross-references between modules

### Development Guides
- **BUILD_DOCS.md** - How to build documentation
- **API_DOCUMENTATION.md** - API documentation overview
- **DOCSTRING_GUIDE.md** - Docstring writing guide

## Writing Documentation

### Adding API Documentation

1. Write NumPy-style docstrings in your code:
   ```python
   def my_function(param: int) -> str:
       """
       Brief description.
       
       Parameters
       ----------
       param : int
           Parameter description
           
       Returns
       -------
       str
           Return value description
       """
   ```

2. Add module to appropriate `.rst` file in `api/`

3. Rebuild documentation

### Docstring Style

Follow the NumPy docstring convention:
- Brief one-line description
- Parameters section with types
- Returns section with type
- Examples section with code
- See `DOCSTRING_GUIDE.md` for details

### Building Locally

```bash
# Clean build
make clean
make html

# Live preview with auto-reload
pip install sphinx-autobuild
sphinx-autobuild . _build/html
# Opens at http://localhost:8000
```

## Documentation Standards

### Quality Checklist

- [ ] All public APIs documented
- [ ] Docstrings follow NumPy style
- [ ] Type hints on all parameters
- [ ] Examples provided where helpful
- [ ] No Sphinx build warnings
- [ ] Links between modules work
- [ ] Code examples are tested

### Testing Documentation

```bash
# Build and check for errors
cd docs
make html

# Check docstring coverage
pip install interrogate
interrogate neural/

# Validate docstrings
pip install pydocstyle
pydocstyle neural/
```

## Continuous Integration

Documentation is automatically built and published:
- On every push to main branch
- Published to Read the Docs (if configured)
- Checked for warnings in CI

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)

## Contributing

When contributing code:

1. **Always add docstrings** to public functions/classes
2. **Follow NumPy style** for consistency
3. **Include type hints** in signatures
4. **Test your examples** to ensure they work
5. **Update .rst files** if adding new modules
6. **Build docs locally** before submitting PR

See `DOCSTRING_GUIDE.md` for detailed guidelines.

## Troubleshooting

### Import Errors
```bash
# Ensure Neural DSL is installed
pip install -e .
```

### Missing Dependencies
```bash
# Install documentation dependencies
pip install -e ".[docs]"
```

### Build Warnings
Check the warning messages and fix:
- Missing docstrings
- Broken cross-references
- Invalid reStructuredText syntax

### Theme Not Found
```bash
pip install sphinx-rtd-theme
```

## Support

For documentation issues:
- Open an issue on GitHub
- Check existing documentation
- Refer to Sphinx documentation

## License

Documentation is released under the same license as Neural DSL (MIT License).
