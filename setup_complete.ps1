# Neural DSL - Complete Setup Script (PowerShell)
# This script creates venv and installs all dependencies

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "Neural DSL - Initial Setup"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment already exists
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists at .venv\" -ForegroundColor Yellow
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created successfully" -ForegroundColor Green
}

Write-Host ""
Write-Host "Installing packages..." -ForegroundColor Green
Write-Host ""

# Install core package
Write-Host "[1/2] Installing Neural DSL package (editable mode)..." -ForegroundColor Cyan
& .venv\Scripts\python.exe -m pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install package" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/2] Installing development dependencies..." -ForegroundColor Cyan
& .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install development dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the virtual environment:" -ForegroundColor Yellow
Write-Host "  PowerShell:  .\.venv\Scripts\Activate.ps1"
Write-Host "  CMD:         .venv\Scripts\activate.bat"
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host "  Lint:       python -m ruff check ."
Write-Host "  Type Check: python -m mypy neural/ --ignore-missing-imports"
Write-Host "  Test:       python -m pytest tests/ -v"
Write-Host ""
