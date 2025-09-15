from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import numpy as np
from typing import List, Dict, Any
from src.visual_persona_generator import VisualPersonaGenerator
from src.pdf_report_generator import PDFReportGenerator
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

app = FastAPI(title="Enhanced Visual Persona Generator", description="AI-powered image analysis with GPU acceleration and Gemini integration")

# Mount templates
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "data/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load configuration from .env file
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 10485760))  # Default 10MB
MAX_FILES = int(os.getenv('MAX_FILES', 50))  # Default 50 files
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', 8000))

# Initialize with Gemini API key from .env file
if GEMINI_API_KEY:
    logger.info("🤖 Gemini API key loaded from .env file - enabling enhanced context analysis")
    logger.info(f"🔑 API Key: {GEMINI_API_KEY[:20]}...")  # Show first 20 chars for verification
else:
    logger.warning("⚠️  Gemini API key not found in .env file - using heuristic analysis only")
    logger.info("💡 Add GEMINI_API_KEY=your_key to .env file for enhanced features")

generator = VisualPersonaGenerator(gemini_api_key=GEMINI_API_KEY)
pdf_generator = PDFReportGenerator()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} images allowed")
    
    uploaded_files = []
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
        if file.size > MAX_FILE_SIZE:  # Use configurable file size
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            raise HTTPException(status_code=400, detail=f"File {file.filename} is too large (max {max_mb}MB)")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)
    
    # Analyze the uploaded images
    uploaded_paths = [os.path.join(UPLOAD_DIR, f) for f in uploaded_files]
    try:
        results = generator.analyze_batch(uploaded_paths)
        report_path = generator.generate_report(results)
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = convert_numpy_types(results)
        
        # Cleanup
        generator.cleanup(uploaded_paths)
        return JSONResponse(content={
            "message": "Analysis complete", 
            "results": serializable_results, 
            "report_path": report_path
        })
    except Exception as e:
        # Cleanup on error
        generator.cleanup(uploaded_paths)
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-pdf")
async def generate_pdf_report(request: Request):
    try:
        data = await request.json()
        results = data.get('results', [])
        
        if not results:
            raise HTTPException(status_code=400, detail="No analysis results provided")
        
        pdf_path = pdf_generator.generate_pdf_report(results)
        
        return FileResponse(
            pdf_path, 
            media_type='application/pdf',
            filename='visual_persona_report.pdf'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Visual Persona Generator API is running"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced Visual Persona Generator...")
    print("📊 Intelligent Emotion Analysis with GPU Acceleration")
    print(f"🌐 Server will be available at: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"🤖 Gemini AI: {'✅ Enabled' if GEMINI_API_KEY else '❌ Disabled (add to .env file)'}")
    print("💡 Note: For development with auto-reload, use: uvicorn app:app --reload")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)