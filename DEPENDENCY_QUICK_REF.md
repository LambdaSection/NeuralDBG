# Dependency Quick Reference Card

## Installation Commands

### Basic Installations
```bash
# Core only (20MB)
pip install neural-dsl

# Full (8GB)
pip install neural-dsl[full]
```

### Feature Groups
```bash
pip install neural-dsl[backends]       # TensorFlow, PyTorch, ONNX
pip install neural-dsl[hpo]            # Optuna, scikit-learn
pip install neural-dsl[visualization]  # matplotlib, graphviz, plotly
pip install neural-dsl[dashboard]      # Dash, Flask
pip install neural-dsl[cloud]          # GitHub, Selenium, Tweepy
pip install neural-dsl[api]            # FastAPI
pip install neural-dsl[utils]          # psutil, pandas, scipy
pip install neural-dsl[ml-extras]      # HuggingFace, Transformers
pip install neural-dsl[dev]            # pytest, ruff, mypy
```

### Multiple Features
```bash
pip install neural-dsl[backends,visualization,hpo]
```

### From Source
```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -e .                # Core only
pip install -e ".[full]"        # All features
```

### Development
```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Common Use Cases

| Use Case | Command |
|----------|---------|
| Learning DSL | `pip install neural-dsl` |
| PyTorch dev | `pip install neural-dsl[backends]` |
| TensorFlow dev | `pip install neural-dsl[backends]` |
| Using dashboard | `pip install neural-dsl[dashboard,visualization]` |
| HPO experiments | `pip install neural-dsl[backends,hpo]` |
| Contributing | `pip install -r requirements-dev.txt` |
| Everything | `pip install neural-dsl[full]` |

## Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | `pip install neural-dsl[backends]` |
| `ModuleNotFoundError: dash` | `pip install neural-dsl[dashboard]` |
| `ModuleNotFoundError: optuna` | `pip install neural-dsl[hpo]` |
| Large install size | `pip install neural-dsl` (core only) |

## Verification
```bash
neural --version
neural --help
neural compile examples/mnist.neural --backend tensorflow
```

## Documentation
- [INSTALL.md](INSTALL.md) - Installation guide
- [DEPENDENCY_GUIDE.md](DEPENDENCY_GUIDE.md) - Complete reference
- [MIGRATION_GUIDE_DEPENDENCIES.md](MIGRATION_GUIDE_DEPENDENCIES.md) - Migration guide

## Quick Links
- Install: [INSTALL.md](INSTALL.md)
- Dependency Info: [DEPENDENCY_GUIDE.md](DEPENDENCY_GUIDE.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Issues: https://github.com/Lemniscate-world/Neural/issues
- Discord: https://discord.gg/KFku4KvS
