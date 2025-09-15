
# 🎭 Visual Persona Generator

AI-powered tool that analyzes your images to generate **visual personality profiles**.  
It detects **faces, emotions, colors, objects, and scenes**, then combines them into **insightful personality & mood reports**.  

---

## ✨ How It Works
- **Face & Emotion Analysis** → Detects happiness, sadness, anger, fear, disgust, surprise  
- **Scene & Context** → Matches emotions with the environment using Gemini AI  
- **Color Psychology** → Reads moods from dominant colors  
- **Objects & Lifestyle** → Identifies activities, lifestyle markers, and context  
- **Reports** → Interactive web charts + downloadable PDF  

---

## ⚡ Quick Start

### Automated Setup
```bash
# Windows
setup_windows.bat

# Linux/Mac
chmod +x setup_unix.sh
./setup_unix.sh
````

### Manual Setup

```bash
git clone https://github.com/yourusername/visual-persona-generator.git
cd visual-persona-generator

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

(Optional) Add your Gemini API key:

```bash
echo GEMINI_API_KEY=your_api_key_here > .env
```

---

## 🚀 Run the App

```bash
python app.py
# or
uvicorn app:app --reload
```

Then open 👉 [http://localhost:8000](http://localhost:8000)

---

## 🛠️ Features at a Glance

* 🎭 Emotion & face analysis (real AI models, GPU support)
* 🌄 Scene & context mood detection
* 🎨 Color psychology insights
* 📊 PDF & web reports with emotion charts
* ⚡ FastAPI backend + simple web frontend

---

## 📌 Example Use

Upload a photo → Get:

* Emotion percentages (😊, 😢, 😠, etc.)
* Scene & color mood match
* Lifestyle/object insights
* Downloadable PDF personality report
