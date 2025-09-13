@echo off
echo ========================================
echo   Visual Persona Generator - Enhanced
echo ========================================
echo.
echo Starting the intelligent emotion analysis system...
echo.
echo Available startup methods:
echo 1. Direct: python app.py (this method)
echo 2. Development: uvicorn app:app --reload
echo.
echo Opening in 3 seconds...
timeout /t 3 /nobreak > nul
echo.
echo 🚀 Starting server...

REM Check if virtual environment exists
if exist ".venv\Scripts\python.exe" (
    echo ✅ Using virtual environment
    ".venv\Scripts\python.exe" app.py
) else (
    echo ⚠️  Virtual environment not found, using system Python
    python app.py
)

echo.
echo Server stopped. Press any key to exit...
pause > nul