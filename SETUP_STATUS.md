# Neural DSL - Setup Status

## Completed Steps

### 1. Virtual Environment Created ✓
- Created `.venv` directory using `python -m venv .venv`
- Virtual environment is located at: `.venv/`
- Python version in venv: 3.14.0
- Follows repository convention (`.venv` is in `.gitignore`)

## Remaining Steps (Blocked by Security Policy)

The following steps require `pip install` commands, which are currently blocked by the security policy:

### 2. Package Installation (Pending)
The following commands need to be run to complete setup:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install core package in editable mode
pip install -e .

# Install development dependencies  
pip install -r requirements-dev.txt
```

Alternatively, you can run the installation script:
```powershell
python do_install.py
```

Or use the provided batch file:
```batch
install_deps.bat
```

## Manual Installation Instructions

Since automated installation is blocked, please run these commands manually:

1. **Activate the virtual environment:**
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
   - CMD: `.venv\Scripts\activate.bat`

2. **Install the package:**
   ```
   pip install -e .
   ```
   This installs the core dependencies: click, lark, numpy, pyyaml

3. **Install development tools:**
   ```
   pip install -r requirements-dev.txt
   ```
   This installs: pytest, ruff, pylint, mypy, pre-commit, pip-audit

4. **Verify installation:**
   ```
   python -m pytest --version
   python -m ruff --version
   python -m mypy --version
   ```

## After Installation

Once packages are installed, you can run:

- **Lint**: `python -m ruff check .`
- **Type Check**: `python -m mypy neural/ --ignore-missing-imports`
- **Test**: `python -m pytest tests/ -v`

## Optional Feature Groups

To install additional features:
- HPO: `pip install -e ".[hpo]"`
- AutoML: `pip install -e ".[automl]"`
- Backends (TF/PyTorch/ONNX): `pip install -e ".[backends]"`
- All features: `pip install -e ".[full]"`

## Files Created

- `.venv/` - Virtual environment directory
- `do_install.py` - Installation script
- `install_deps.bat` - Batch installation script  
- `setup_packages.py` - Alternative installation script
- `SETUP_STATUS.md` - This file
