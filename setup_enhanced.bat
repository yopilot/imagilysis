@echo off
echo.
echo ========================================
echo  Enhanced Visual Persona Generator Setup
echo ========================================
echo.

echo Installing enhanced dependencies...
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python...
)

echo.
echo Installing core dependencies...
pip install facenet-pytorch timm google-generativeai

echo.
echo Installing all requirements...
pip install -r requirements.txt

echo.
echo Checking for GPU support...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print('GPU Names:'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('  No GPU detected')"

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo To use Gemini API (optional but recommended):
echo 1. Get API key from: https://aistudio.google.com/
echo 2. Set environment variable: set GEMINI_API_KEY=your_key_here
echo 3. Restart the server
echo.
echo Start server with:
echo   python app.py
echo   OR
echo   start_server.bat
echo.
echo Open browser to: http://localhost:8000
echo.
pause