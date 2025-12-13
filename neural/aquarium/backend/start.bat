@echo off
REM Start Neural Aquarium Backend API

echo Starting Neural Aquarium Backend API...
echo API will be available at http://localhost:5000
echo.

cd /d "%~dp0"

if not exist "..\..\..\\.venv" if not exist "..\\..\\..\\venv" (
    echo Warning: Virtual environment not found. Installing dependencies globally...
    pip install -r requirements.txt
) else (
    if exist "..\..\..\\.venv" (
        call ..\..\..\\.venv\Scripts\activate.bat
    ) else (
        call ..\\..\\..\\venv\Scripts\activate.bat
    )
    echo Virtual environment activated
)

set FLASK_APP=api.py
set FLASK_ENV=development

python api.py
