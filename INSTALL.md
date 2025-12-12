# Neural DSL Installation Quick Reference

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Quick Install Commands

### For End Users

```bash
# Minimal installation (core DSL only)
pip install neural-dsl

# With ML backends (PyTorch, TensorFlow, ONNX)
pip install neural-dsl[backends]

# With visualization tools
pip install neural-dsl[visualization]

# With NeuralDbg dashboard
pip install neural-dsl[dashboard]

# With HPO support
pip install neural-dsl[hpo]

# With cloud integrations
pip install neural-dsl[cloud]

# Complete installation (all features)
pip install neural-dsl[full]
```

### For Developers

```bash
# Clone repository
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

## Installation Profiles

### Profile 1: Minimal (Learning/DSL Only)
**Size**: ~20 MB  
**Use case**: Learning DSL syntax, generating code without execution

```bash
pip install neural-dsl
```

### Profile 2: PyTorch Developer
**Size**: ~3-4 GB  
**Use case**: Training PyTorch models

```bash
pip install neural-dsl[backends,visualization]
```

### Profile 3: TensorFlow Developer
**Size**: ~2-3 GB  
**Use case**: Training TensorFlow models

```bash
pip install neural-dsl
pip install tensorflow matplotlib
```

### Profile 4: Research/HPO
**Size**: ~5-6 GB  
**Use case**: Hyperparameter optimization experiments

```bash
pip install neural-dsl[backends,hpo,visualization]
```

### Profile 5: Full Stack Developer
**Size**: ~8 GB  
**Use case**: Using all features including dashboard and cloud

```bash
pip install neural-dsl[full]
```

### Profile 6: Contributor
**Size**: ~8 GB + dev tools  
**Use case**: Contributing to Neural DSL development

```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -r requirements-dev.txt
pre-commit install
```

## Alternative Requirements Files

For convenience, you can use specific requirements files:

```bash
# Minimal core only
pip install -r requirements-minimal.txt

# Core + backends
pip install -r requirements-backends.txt

# Core + visualization
pip install -r requirements-viz.txt

# Development setup
pip install -r requirements-dev.txt
```

## Verify Installation

After installation, verify Neural DSL is working:

```bash
# Check version
neural --version

# Run help
neural --help

# Test with example
neural compile examples/mnist.neural --backend tensorflow
```

## Upgrade

```bash
# Upgrade to latest version
pip install --upgrade neural-dsl

# Upgrade with specific features
pip install --upgrade neural-dsl[full]
```

## Uninstall

```bash
pip uninstall neural-dsl
```

## Troubleshooting

### Issue: Import errors for torch/tensorflow
**Solution**: Install the backends group
```bash
pip install neural-dsl[backends]
```

### Issue: Import errors for dash/flask
**Solution**: Install the dashboard group
```bash
pip install neural-dsl[dashboard]
```

### Issue: Visualization commands fail
**Solution**: Install visualization dependencies
```bash
pip install neural-dsl[visualization]
```

### Issue: Large installation size
**Solution**: Install only what you need
```bash
# Instead of [full], use specific groups
pip install neural-dsl[backends,visualization]
```

### Issue: Conflicting versions
**Solution**: Use fresh virtual environment
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
pip install neural-dsl[full]
```

## Platform-Specific Notes

### Windows
- Use `.venv\Scripts\activate` to activate virtual environment
- May need Visual C++ Build Tools for some dependencies
- Administrator privileges might be needed for graphviz

### Linux
- May need to install graphviz system package: `sudo apt-get install graphviz`
- Some distributions require python3-venv: `sudo apt-get install python3-venv`

### macOS
- Install graphviz via Homebrew: `brew install graphviz`
- May need Xcode Command Line Tools: `xcode-select --install`

## Docker

For containerized environments:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y graphviz

# Install Neural DSL
RUN pip install neural-dsl[full]

# Verify installation
RUN neural --version
```

## Cloud Notebooks

### Google Colab
```python
!pip install neural-dsl[backends,visualization,cloud]
```

### Kaggle
```python
!pip install neural-dsl[backends,visualization,cloud]
```

### AWS SageMaker
```python
!pip install neural-dsl[full]
```

## Support

- Documentation: [README.md](README.md)
- Dependency Guide: [DEPENDENCY_GUIDE.md](DEPENDENCY_GUIDE.md)
- Issues: https://github.com/Lemniscate-world/Neural/issues
- Discord: https://discord.gg/KFku4KvS
