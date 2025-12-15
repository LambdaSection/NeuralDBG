#!/usr/bin/env python
"""Helper script to install dependencies in the virtual environment."""
import subprocess
import sys
from pathlib import Path

# Get the venv python executable
venv_python = Path(".venv") / "Scripts" / "python.exe"

if not venv_python.exists():
    print("Error: Virtual environment not found at .venv")
    sys.exit(1)

print(f"Using Python: {venv_python}")

# Install the package in editable mode
print("\n=== Installing neural-dsl in editable mode ===")
result = subprocess.run([str(venv_python), "-m", "pip", "install", "-e", "."], check=False)
if result.returncode != 0:
    print("Error installing neural-dsl")
    sys.exit(1)

print("\n=== Installing development dependencies ===")
result = subprocess.run([str(venv_python), "-m", "pip", "install", "-r", "requirements-dev.txt"], check=False)
if result.returncode != 0:
    print("Error installing development dependencies")
    sys.exit(1)

print("\n=== Setup complete! ===")
print(f"To activate the virtual environment, run: .venv\\Scripts\\Activate.ps1")
