# 🚀 QUICK START GUIDE

## Enhanced Visual Persona Generator - Get Started in 2 Minutes!

### 📋 Before You Begin
- **Python 3.11+** installed ([Download here](https://www.python.org/downloads/))
- **5GB free disk space** for AI models
- **Internet connection** for downloads

### ⚡ Automated Setup (Recommended)

#### Windows Users:
```bash
# Double-click this file or run in terminal:
setup_windows.bat
```

#### Linux/Mac Users:
```bash
# Run in terminal:
./setup_unix.sh
```

#### Manual (All platforms):
```bash
python setup.py
```

### 🎯 What Happens During Setup:
1. **Creates virtual environment** (isolated Python environment)
2. **Installs all dependencies** (FastAPI, PyTorch, Transformers, etc.)
3. **Downloads AI models** (MTCNN, HuggingFace emotion models, DETR)
4. **Sets up GPU acceleration** (if NVIDIA GPU available)
5. **Creates configuration files** (.env, directories)
6. **Tests everything** to ensure it works

### ⏱️ Setup Time:
- **5-10 minutes** with fast internet
- **10-15 minutes** with slower connection
- **First-time only** - subsequent starts are instant!

### 🔑 After Setup:
1. **Add Gemini API Key** (optional but recommended):
   - Get key from: https://aistudio.google.com/app/apikey
   - Add to `.env` file: `GEMINI_API_KEY=your_key_here`

2. **Start the application**:
   ```bash
   # Windows
   .venv\Scripts\python.exe app.py
   
   # Linux/Mac  
   .venv/bin/python app.py
   ```

3. **Open browser**: http://localhost:8000

### 🎉 You're Ready!
- Upload images and get instant AI analysis
- Real emotion detection with 6 core emotions
- Beautiful visual reports and PDF generation
- GPU acceleration for maximum speed

### 🆘 Need Help?
- Check `setup_log.txt` for detailed setup information
- See `README.md` for comprehensive documentation
- All models download automatically if setup script fails

---
**Built with ❤️ for instant AI-powered image analysis**