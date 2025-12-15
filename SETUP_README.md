# Neural DSL - Setup Complete

## What Has Been Done

✅ **Virtual Environment Created**
- Location: `.venv/`
- Python Version: 3.14.0
- Configuration: `.venv/pyvenv.cfg` exists

## What Needs To Be Done

⚠️ **Package Installation Required**

Due to security restrictions in the automated environment, the package installation step needs to be completed manually. The virtual environment is ready, but packages have not been installed yet.

## Complete Setup - Choose One Method

### Method 1: Automated Script (Recommended)

**Windows (PowerShell):**
```powershell
.\setup_complete.ps1
```

**Windows (CMD):**
```batch
setup_complete.bat
```

### Method 2: Manual Installation

1. **Activate the virtual environment:**

   **PowerShell:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **CMD:**
   ```batch
   .venv\Scripts\activate.bat
   ```

2. **Install the package and dependencies:**
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

### Method 3: Using Makefile (If make is available)

```bash
make install-dev
```

## Verify Installation

After running one of the above methods, verify the installation:

```bash
# Activate venv first
.\.venv\Scripts\Activate.ps1  # PowerShell
# or
.venv\Scripts\activate.bat    # CMD

# Then verify
python -m pytest --version
python -m ruff --version
python -m mypy --version
neural --help
```

## Core Dependencies Installed

After setup, you will have:

**Core Package:**
- click >= 8.1.3
- lark >= 1.1.5
- numpy >= 1.23.0
- pyyaml >= 6.0.1

**Development Tools:**
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- ruff >= 0.1.0
- pylint >= 2.15.0
- mypy >= 1.0.0
- pre-commit >= 3.0.0
- pip-audit >= 2.0.0

## Running Build, Lint, and Tests

Once installation is complete, you can run:

### Lint
```bash
python -m ruff check .
```

### Type Check
```bash
python -m mypy neural/ --ignore-missing-imports
```

### Tests
```bash
python -m pytest tests/ -v
```

Or with coverage:
```bash
pytest --cov=neural --cov-report=term
```

## Optional Features

To install additional feature groups:

```bash
# Hyperparameter optimization
pip install -e ".[hpo]"

# AutoML and NAS
pip install -e ".[automl]"

# ML Backends (TensorFlow, PyTorch, ONNX)
pip install -e ".[backends]"

# Visualization tools
pip install -e ".[visualization]"

# Dashboard (NeuralDbg)
pip install -e ".[dashboard]"

# All features
pip install -e ".[full]"
```

## Repository Structure

- `neural/` - Main package source code
- `tests/` - Test suite
- `examples/` - Example DSL files
- `docs/` - Documentation
- `.venv/` - Virtual environment (created)
- `setup.py` - Package configuration
- `pyproject.toml` - Build configuration
- `requirements-dev.txt` - Development dependencies

## Next Steps

1. Run one of the installation methods above
2. Verify installation works
3. Try running lint, tests, and type checks
4. Start developing!

## Troubleshooting

### Virtual Environment Not Activating
- Ensure you're using the correct path separator for your shell
- PowerShell: `.\.venv\Scripts\Activate.ps1`
- CMD: `.venv\Scripts\activate.bat`

### Package Installation Fails
- Ensure pip is up to date: `python -m pip install --upgrade pip`
- Try installing dependencies individually if there are version conflicts

### Import Errors
- Make sure the virtual environment is activated
- Verify installation with `pip list`

## Support

For more information, see:
- `AGENTS.md` - Development guide
- `ARCHITECTURE.md` - System architecture
- `CONTRIBUTING.md` - Contribution guidelines
- `INSTALL.md` - Detailed installation guide
