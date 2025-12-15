@echo off
REM Neural DSL - Complete Setup Script
REM This script creates venv and installs all dependencies

echo ========================================
echo Neural DSL - Initial Setup
echo ========================================
echo.

REM Check if virtual environment already exists
if exist .venv\ (
    echo Virtual environment already exists at .venv\
) else (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        exit /b 1
    )
    echo Virtual environment created successfully
)

echo.
echo Installing packages...
echo.

REM Install core package
echo [1/2] Installing Neural DSL package (editable mode)...
.venv\Scripts\python.exe -m pip install -e .
if errorlevel 1 (
    echo Error: Failed to install package
    exit /b 1
)

echo.
echo [2/2] Installing development dependencies...
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if errorlevel 1 (
    echo Error: Failed to install development dependencies
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate the virtual environment:
echo   PowerShell:  .\.venv\Scripts\Activate.ps1
echo   CMD:         .venv\Scripts\activate.bat
echo.
echo Available commands:
echo   Lint:       python -m ruff check .
echo   Type Check: python -m mypy neural/ --ignore-missing-imports
echo   Test:       python -m pytest tests/ -v
echo.
