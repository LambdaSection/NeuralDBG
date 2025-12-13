@echo off
REM Start both frontend and backend for development

echo Starting Neural Aquarium Development Environment
echo ================================================
echo.

cd /d "%~dp0"

echo Starting Backend API...
start "Neural Aquarium Backend" cmd /k "cd backend && start.bat"

timeout /t 3 /nobreak >nul

echo.
echo Starting Frontend...
start "Neural Aquarium Frontend" cmd /k "npm start"

echo.
echo Development servers started:
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:5000
echo.
echo Press any key to exit...
pause >nul
