# Neural DSL Dependency Management Guide

This guide explains Neural DSL's modular dependency structure and how to manage dependencies effectively.

## Philosophy

Neural DSL uses a **minimal core + optional features** approach to:
- Keep the base installation lightweight
- Allow users to install only what they need
- Reduce installation time and disk space
- Avoid forcing heavy dependencies (TensorFlow, PyTorch) on users who don't need them

## Installation Options

### 1. Minimal Installation (Core Only)

**Use case**: You only need DSL parsing, compilation, and basic CLI functionality.

```bash
pip install neural-dsl
```

**Includes**:
- `click>=8.1.3` - CLI framework
- `lark>=1.1.5` - DSL parser
- `numpy>=1.23.0` - Numerical operations
- `pyyaml>=6.0.1` - Configuration parsing

**Size**: ~20 MB

### 2. Feature-Specific Installation

Install only the features you need:

#### ML Framework Backends
```bash
pip install neural-dsl[backends]
```
Includes: TensorFlow, PyTorch, ONNX, torchvision, onnxruntime

#### Hyperparameter Optimization
```bash
pip install neural-dsl[hpo]
```
Includes: Optuna, scikit-learn

#### Visualization Tools
```bash
pip install neural-dsl[visualization]
```
Includes: matplotlib, graphviz, networkx, plotly, seaborn

#### NeuralDbg Dashboard
```bash
pip install neural-dsl[dashboard]
```
Includes: Dash, Flask, Flask-CORS, Flask-HTTPAuth, Flask-SocketIO, plotly

#### Cloud Integrations
```bash
pip install neural-dsl[cloud]
```
Includes: pygithub, selenium, webdriver-manager, tweepy

#### API Server Support
```bash
pip install neural-dsl[api]
```
Includes: FastAPI

#### Utility Tools
```bash
pip install neural-dsl[utils]
```
Includes: psutil, pysnooper, radon, pandas, scipy, statsmodels, sympy, multiprocess

#### ML Extras (HuggingFace, Transformers)
```bash
pip install neural-dsl[ml-extras]
```
Includes: huggingface_hub, transformers

### 3. Combined Installation

Install multiple feature groups:

```bash
# Backends + Visualization + HPO
pip install neural-dsl[backends,visualization,hpo]

# Everything except cloud and API
pip install neural-dsl[backends,hpo,visualization,dashboard,utils,ml-extras]
```

### 4. Full Installation

**Use case**: You want all features for development or comprehensive usage.

```bash
pip install neural-dsl[full]
```

**Size**: ~5-8 GB (due to TensorFlow and PyTorch)

### 5. Development Installation

**Use case**: You're contributing to Neural DSL.

```bash
# Clone the repository
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

**Includes**:
- Core dependencies
- Testing tools (pytest, pytest-cov)
- Linting tools (ruff, pylint, flake8)
- Type checking (mypy)
- Git hooks (pre-commit)
- Security auditing (pip-audit)

## Dependency Files

### `requirements.txt`
Contains **only core dependencies**. Used for minimal installations.

```bash
pip install -r requirements.txt
```

### `requirements-dev.txt`
Contains development dependencies plus core dependencies. Used by contributors.

```bash
pip install -r requirements-dev.txt
```

### `setup.py`
Defines all dependency groups and installation options. Source of truth for package dependencies.

## Use Cases and Recommendations

### Use Case 1: Learning the DSL Syntax
**Installation**: Minimal
```bash
pip install neural-dsl
```

### Use Case 2: Training PyTorch Models
**Installation**: Core + Backends
```bash
pip install neural-dsl[backends]
```

### Use Case 3: Using NeuralDbg Dashboard
**Installation**: Core + Dashboard + Visualization
```bash
pip install neural-dsl[dashboard,visualization]
```

### Use Case 4: HPO Experiments
**Installation**: Core + Backends + HPO
```bash
pip install neural-dsl[backends,hpo]
```

### Use Case 5: Cloud Notebook (Kaggle, Colab)
**Installation**: Core + Backends + Cloud + Visualization
```bash
pip install neural-dsl[backends,cloud,visualization]
```

### Use Case 6: Contributing to Neural DSL
**Installation**: Development setup
```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -r requirements-dev.txt
pre-commit install
```

## Individual Package Installation

You can also install specific packages manually:

```bash
# Just PyTorch (no TensorFlow)
pip install neural-dsl
pip install torch torchvision

# Just TensorFlow (no PyTorch)
pip install neural-dsl
pip install tensorflow

# Just visualization
pip install neural-dsl
pip install matplotlib graphviz plotly

# Just HPO
pip install neural-dsl
pip install optuna scikit-learn
```

## CI/CD Considerations

### GitHub Actions Workflows

For testing in CI, use targeted installations:

```yaml
# Minimal tests (no ML backends)
- name: Install dependencies
  run: pip install -e .

# Backend tests
- name: Install dependencies
  run: pip install -e ".[backends]"

# Full test suite
- name: Install dependencies
  run: pip install -e ".[full]"
```

### Docker Images

Layer your Docker builds for better caching:

```dockerfile
# Stage 1: Core dependencies
RUN pip install neural-dsl

# Stage 2: Add backends
RUN pip install neural-dsl[backends]

# Stage 3: Add visualization
RUN pip install neural-dsl[visualization]
```

## Troubleshooting

### Import Errors

If you get import errors like:
```
ModuleNotFoundError: No module named 'torch'
```

Install the missing feature group:
```bash
pip install neural-dsl[backends]
```

### Conflicting Dependencies

If you encounter version conflicts:

1. Use a fresh virtual environment
2. Install Neural DSL first
3. Install other packages after

```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install neural-dsl[full]
pip install your-other-package
```

### Large Installation Size

If disk space is limited:

1. Install only core dependencies
2. Install individual backends as needed
3. Avoid `[full]` installation

```bash
pip install neural-dsl          # Core only (~20 MB)
pip install torch               # Add PyTorch when needed
```

## Migration from Old Setup

If you previously installed with `requirements.txt` (which had all dependencies):

### Old Way (All Dependencies)
```bash
pip install -r requirements.txt  # ~8 GB
```

### New Way (Modular)
```bash
# Option 1: Minimal
pip install neural-dsl  # ~20 MB

# Option 2: Specific features
pip install neural-dsl[backends,visualization]  # ~5-6 GB

# Option 3: Everything (equivalent to old way)
pip install neural-dsl[full]  # ~8 GB
```

## Contributing to Dependency Management

When adding new dependencies:

1. **Categorize correctly**: Place in the appropriate feature group
2. **Update setup.py**: Add to the relevant `*_DEPS` list
3. **Document use case**: Explain why the dependency is needed
4. **Test minimal installation**: Ensure core functionality still works with just core deps
5. **Update this guide**: Document new feature groups or dependencies

### Adding a New Feature Group

Example: Adding a new "database" feature group:

```python
# In setup.py
DATABASE_DEPS = [
    "sqlalchemy>=2.0",
    "psycopg2-binary>=2.9",
]

extras_require={
    "database": DATABASE_DEPS,
    "full": (
        CORE_DEPS
        + DATABASE_DEPS  # Add to full
        # ... other deps
    ),
}
```

## Version Compatibility

Neural DSL maintains compatibility with:
- **Python**: 3.8, 3.9, 3.10, 3.11
- **TensorFlow**: 2.6+
- **PyTorch**: 1.10+
- **ONNX**: 1.10+

Check `setup.py` for specific version constraints.

## Best Practices

1. **Use virtual environments**: Always isolate Neural DSL installations
2. **Pin versions in production**: Use exact versions in production deployments
3. **Regular updates**: Keep dependencies updated for security
4. **Audit dependencies**: Run `pip-audit` to check for vulnerabilities
5. **Document choices**: Note which feature groups you're using in your project docs

## Support

For dependency-related issues:
- Check [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- Ask in [Discord](https://discord.gg/KFku4KvS)
- Consult [README.md](README.md) for installation examples
