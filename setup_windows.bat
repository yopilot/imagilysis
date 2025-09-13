@echo off
REM Enhanced Visual Persona Generator - Windows Setup Launcher
REM ===========================================================
REM This batch file launches the Python setup script for easy installation

echo.
echo ================================================================
echo  Enhanced Visual Persona Generator - Automated Setup (Windows)
echo ================================================================
echo.
echo This will automatically set up everything needed:
echo  * Python virtual environment
echo  * All dependencies with GPU acceleration
echo  * AI model downloads and caching
echo  * Complete configuration and testing
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found in PATH
    echo.
    echo Please install Python 3.11+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Display Python version
echo 🐍 Python version:
python --version
echo.

REM Check Python version requirement
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>nul
if errorlevel 1 (
    echo ❌ ERROR: Python 3.11+ required
    echo.
    echo Please update Python from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Run the setup script
echo 🚀 Launching automated setup...
echo.
python setup.py

REM Check if setup was successful
if errorlevel 1 (
    echo.
    echo ❌ Setup encountered issues. Check setup_log.txt for details.
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ✅ Setup completed! You can now run the application.
    echo.
    echo Quick start commands:
    echo   1. Start server: .venv\Scripts\python.exe app.py
    echo   2. Open browser: http://localhost:8000
    echo.
    echo Press any key to exit...
    pause >nul
)