@echo off
REM Script to run Aquarium E2E tests on Windows

setlocal enabledelayedexpansion

echo ======================================================================
echo Aquarium IDE End-to-End Tests
echo ======================================================================

REM Check if Playwright is installed
python -c "import playwright" 2>nul
if errorlevel 1 (
    echo Error: Playwright is not installed
    echo Install with: pip install playwright ^&^& playwright install chromium
    exit /b 1
)

REM Check if Aquarium server dependencies are available
python -c "import dash" 2>nul
if errorlevel 1 (
    echo Warning: Dash not installed. Some tests may fail.
    echo Install with: pip install -e .[dashboard]
)

REM Run tests
cd /d "%~dp0"

if "%1"=="--help" goto :help
if "%1"=="-h" goto :help
if "%1"=="/?" goto :help

python run_tests.py %*
exit /b %errorlevel%

:help
echo Usage: run_tests.bat [OPTIONS]
echo.
echo Options:
echo   --fast        Skip slow tests
echo   --visible     Run with visible browser
echo   --debug       Run in debug mode
echo   --parallel    Run tests in parallel
echo   --help        Show this help message
exit /b 0
