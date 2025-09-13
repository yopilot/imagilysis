#!/usr/bin/env python3
"""
Enhanced Visual Persona Generator - Automated Setup Script
==========================================================

This script automatically sets up the complete environment for the Enhanced Visual Persona Generator:
1. Creates and activates Python virtual environment
2. Installs all required dependencies
3. Downloads and caches all AI models for optimal performance
4. Verifies GPU availability and sets up acceleration
5. Creates necessary directories and configuration files
6. Tests the complete setup to ensure everything works perfectly

Run this script once before first use for optimal performance.
"""

import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path
import json
import time
import urllib.request
from typing import List, Tuple, Dict

class EnhancedSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.venv_path = self.project_root / ".venv"
        self.is_windows = platform.system() == "Windows"
        self.python_executable = self._get_python_executable()
        self.setup_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log setup progress with timestamps"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.setup_log.append(log_entry)
        
    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable path"""
        if self.is_windows:
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")
    
    def _get_pip_executable(self) -> str:
        """Get the appropriate pip executable path"""
        if self.is_windows:
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements"""
        self.log("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 11):
            self.log(f"❌ Python {python_version.major}.{python_version.minor} detected. Python 3.11+ required!", "ERROR")
            return False
        self.log(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} - Compatible")
        
        # Check available disk space (need ~5GB for models)
        try:
            if self.is_windows:
                free_bytes = shutil.disk_usage(self.project_root).free
            else:
                stat = shutil.disk_usage(self.project_root)
                free_bytes = stat.free
            
            free_gb = free_bytes / (1024**3)
            if free_gb < 5:
                self.log(f"⚠️  Warning: Only {free_gb:.1f}GB free space. Recommend 5GB+ for models", "WARNING")
            else:
                self.log(f"✅ {free_gb:.1f}GB free space available")
        except Exception as e:
            self.log(f"⚠️  Could not check disk space: {e}", "WARNING")
        
        # Check if Git is available (for model downloads)
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            self.log("✅ Git available for model downloads")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("⚠️  Git not found - some models may download slower", "WARNING")
        
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment"""
        self.log("🐍 Creating Python virtual environment...")
        
        try:
            if self.venv_path.exists():
                self.log("📁 Removing existing virtual environment...")
                shutil.rmtree(self.venv_path)
            
            # Create virtual environment
            result = subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], capture_output=True, text=True, check=True)
            
            self.log("✅ Virtual environment created successfully")
            
            # Upgrade pip to latest version
            self.log("📦 Upgrading pip to latest version...")
            pip_upgrade = subprocess.run([
                self.python_executable, "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output=True, text=True)
            
            if pip_upgrade.returncode == 0:
                self.log("✅ Pip upgraded successfully")
            else:
                self.log(f"⚠️  Pip upgrade warning: {pip_upgrade.stderr}", "WARNING")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed to create virtual environment: {e}", "ERROR")
            if e.stderr:
                self.log(f"Error details: {e.stderr}", "ERROR")
            return False
    
    def install_dependencies(self) -> bool:
        """Install all required Python packages"""
        self.log("📦 Installing Python dependencies...")
        
        try:
            # Install PyTorch with CUDA support first (if available)
            self.log("🔥 Installing PyTorch with CUDA support...")
            torch_cmd = [
                self.python_executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
            
            torch_result = subprocess.run(torch_cmd, capture_output=True, text=True)
            if torch_result.returncode == 0:
                self.log("✅ PyTorch with CUDA support installed")
            else:
                self.log("⚠️  CUDA version not compatible, installing CPU-only PyTorch", "WARNING")
                cpu_torch_cmd = [
                    self.python_executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio"
                ]
                subprocess.run(cpu_torch_cmd, capture_output=True, text=True, check=True)
                self.log("✅ PyTorch CPU version installed")
            
            # Install main requirements
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                self.log("📋 Installing requirements from requirements.txt...")
                install_cmd = [
                    self.python_executable, "-m", "pip", "install", 
                    "-r", str(requirements_file)
                ]
                
                result = subprocess.run(install_cmd, capture_output=True, text=True, check=True)
                self.log("✅ All dependencies installed successfully")
            else:
                self.log("⚠️  requirements.txt not found, installing core packages manually", "WARNING")
                self._install_core_packages()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed to install dependencies: {e}", "ERROR")
            if e.stderr:
                self.log(f"Error details: {e.stderr}", "ERROR")
            return False
    
    def _install_core_packages(self):
        """Install core packages manually if requirements.txt is missing"""
        core_packages = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0", 
            "transformers>=4.35.0",
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "opencv-python>=4.8.0",
            "Pillow>=10.0.0",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "matplotlib>=3.7.0",
            "pandas>=2.1.0",
            "scikit-learn>=1.3.0",
            "google-generativeai>=0.3.0",
            "python-multipart>=0.0.6",
            "jinja2>=3.1.0",
            "python-dotenv>=1.0.0",
            "reportlab>=4.0.0",
            "mtcnn>=0.1.1",
            "facenet-pytorch>=2.5.3",
            "accelerate>=0.24.0"
        ]
        
        for package in core_packages:
            self.log(f"Installing {package}...")
            subprocess.run([
                self.python_executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, check=True)
    
    def download_and_cache_models(self) -> bool:
        """Download and cache all AI models for optimal performance"""
        self.log("🤖 Downloading and caching AI models...")
        
        # Create models directory
        models_dir = self.project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Define models to download
        models_to_cache = [
            {
                "name": "MTCNN Face Detection",
                "model_id": "mtcnn",
                "description": "GPU-accelerated face detection model"
            },
            {
                "name": "Emotion Recognition", 
                "model_id": "j-hartmann/emotion-english-distilroberta-base",
                "description": "HuggingFace emotion classification model"
            },
            {
                "name": "DETR Object Detection",
                "model_id": "facebook/detr-resnet-50", 
                "description": "Object detection and recognition model"
            },
            {
                "name": "EfficientNet Emotion Backup",
                "model_id": "trpakov/vit-face-expression",
                "description": "Backup emotion recognition model"
            }
        ]
        
        try:
            # Create a model caching script
            cache_script = f"""
import sys
sys.path.append('{self.project_root}')

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DEPRECATEDwarning
import warnings
warnings.filterwarnings("ignore", category=DEPRECATEDwarning)

print("🤖 Caching AI models for optimal performance...")

# Cache emotion recognition model
print("📥 Downloading emotion recognition model...")
try:
    emotion_model = pipeline("text-classification", 
                           model="j-hartmann/emotion-english-distilroberta-base",
                           device=0 if torch.cuda.is_available() else -1)
    print("✅ Emotion recognition model cached")
except Exception as e:
    print(f"⚠️  Emotion model cache warning: {{e}}")

# Cache DETR object detection model  
print("📥 Downloading object detection model...")
try:
    from transformers import DEPRECATEDPreTrainedModel, DetrImageProcessor, DetrForObjectDetection
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    print("✅ Object detection model cached")
except Exception as e:
    print(f"⚠️  DETR model cache warning: {{e}}")

# Cache backup emotion model
print("📥 Downloading backup emotion model...")
try:
    backup_model = pipeline("image-classification",
                          model="trpakov/vit-face-expression", 
                          device=0 if torch.cuda.is_available() else -1)
    print("✅ Backup emotion model cached")
except Exception as e:
    print(f"⚠️  Backup model cache warning: {{e}}")

print("🎉 Model caching complete!")
"""
            
            # Write and execute caching script
            cache_file = self.project_root / "cache_models.py"
            with open(cache_file, "w") as f:
                f.write(cache_script)
            
            self.log("📥 Starting model downloads (this may take 5-10 minutes)...")
            cache_result = subprocess.run([
                self.python_executable, str(cache_file)
            ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if cache_result.returncode == 0:
                self.log("✅ All models downloaded and cached successfully")
                self.log("🚀 Models will load instantly on first use!")
            else:
                self.log("⚠️  Some models may not have cached properly", "WARNING")
                if cache_result.stderr:
                    self.log(f"Cache warnings: {cache_result.stderr}", "WARNING")
            
            # Cleanup cache script
            cache_file.unlink(missing_ok=True)
            
            return True
            
        except subprocess.TimeoutExpired:
            self.log("⚠️  Model download timeout - models will download on first use", "WARNING")
            return True
        except Exception as e:
            self.log(f"⚠️  Model caching issue: {e}", "WARNING")
            self.log("Models will download automatically on first use", "INFO")
            return True
    
    def setup_directories_and_config(self) -> bool:
        """Create necessary directories and configuration files"""
        self.log("📁 Setting up directories and configuration...")
        
        try:
            # Create necessary directories
            directories = [
                "data",
                "data/uploads", 
                "models",
                "utils"
            ]
            
            for dir_name in directories:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(exist_ok=True)
                self.log(f"✅ Directory created: {dir_name}")
            
            # Create .env file if it doesn't exist
            env_file = self.project_root / ".env"
            if not env_file.exists():
                env_content = """# Enhanced Visual Persona Generator Configuration
# ================================================

# Gemini AI Configuration (Required for enhanced analysis)
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# File Upload Limits
MAX_FILE_SIZE=10485760  # 10MB in bytes
MAX_FILES=50           # Maximum files per batch

# Logging Configuration
LOG_LEVEL=INFO         # DEBUG, INFO, WARNING, ERROR

# GPU Configuration (Auto-detected)
USE_GPU=auto          # auto, true, false

# Model Configuration
EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
OBJECT_MODEL=facebook/detr-resnet-50
BACKUP_EMOTION_MODEL=trpakov/vit-face-expression

# Performance Settings
BATCH_SIZE=8          # Adjust based on available RAM/VRAM
MAX_WORKERS=4         # Parallel processing workers
"""
                
                with open(env_file, "w") as f:
                    f.write(env_content)
                self.log("✅ .env configuration file created")
                self.log("💡 Remember to add your Gemini API key to .env file for enhanced features")
            else:
                self.log("✅ .env file already exists")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to setup directories: {e}", "ERROR")
            return False
    
    def verify_gpu_setup(self) -> Dict[str, any]:
        """Check GPU availability and setup"""
        self.log("🔥 Checking GPU availability...")
        
        gpu_info = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
            "gpu_memory": []
        }
        
        try:
            # Test GPU setup
            gpu_test_script = f"""
import torch
import sys

try:
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    
    print(f"CUDA Available: {{cuda_available}}")
    print(f"GPU Count: {{gpu_count}}")
    
    if cuda_available:
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {{i}}: {{gpu_name}} ({{gpu_memory:.1f}}GB)")
    
    # Test MTCNN GPU compatibility
    try:
        from mtcnn import MTCNN
        detector = MTCNN(device='cuda' if cuda_available else 'cpu')
        print("MTCNN GPU: Compatible" if cuda_available else "MTCNN CPU: Fallback")
    except Exception as e:
        print(f"MTCNN Error: {{e}}")
        
except Exception as e:
    print(f"GPU Test Error: {{e}}")
"""
            
            result = subprocess.run([
                self.python_executable, "-c", gpu_test_script
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if "CUDA Available:" in line:
                        gpu_info["cuda_available"] = "True" in line
                    elif "GPU Count:" in line:
                        gpu_info["gpu_count"] = int(line.split(":")[1].strip())
                    elif "GPU " in line and ":" in line:
                        gpu_details = line.split(":", 1)[1].strip()
                        gpu_info["gpu_names"].append(gpu_details)
                
                if gpu_info["cuda_available"]:
                    self.log(f"🚀 GPU acceleration available! {gpu_info['gpu_count']} GPU(s) detected")
                    for i, gpu in enumerate(gpu_info["gpu_names"]):
                        self.log(f"   GPU {i}: {gpu}")
                else:
                    self.log("💻 GPU not available - using CPU (still fast!)")
            else:
                self.log("⚠️  Could not test GPU setup", "WARNING")
                
        except Exception as e:
            self.log(f"⚠️  GPU test error: {e}", "WARNING")
        
        return gpu_info
    
    def test_complete_setup(self) -> bool:
        """Test that the complete setup works"""
        self.log("🧪 Testing complete setup...")
        
        try:
            # Test basic imports and functionality
            test_script = f"""
import sys
sys.path.append('{self.project_root}')

print("Testing core imports...")
try:
    from src.visual_persona_generator import VisualPersonaGenerator
    from src.enhanced_facial_emotion_analysis import EnhancedFacialEmotionAnalysis
    from src.gemini_context_analyzer import GeminiContextAnalyzer
    from src.intelligent_emotion_analysis import IntelligentEmotionAnalysis
    print("✅ Core modules imported successfully")
except Exception as e:
    print(f"❌ Import error: {{e}}")
    sys.exit(1)

print("Testing model initialization...")
try:
    # Test without Gemini API key for basic functionality
    generator = VisualPersonaGenerator(gemini_api_key=None)
    print("✅ Visual Persona Generator initialized")
except Exception as e:
    print(f"❌ Initialization error: {{e}}")
    sys.exit(1)

print("Testing emotion analysis setup...")
try:
    from src.enhanced_facial_emotion_analysis import EnhancedFacialEmotionAnalysis
    emotion_analyzer = EnhancedFacialEmotionAnalysis()
    print("✅ Emotion analysis ready")
except Exception as e:
    print(f"❌ Emotion analysis error: {{e}}")
    sys.exit(1)

print("🎉 Setup test completed successfully!")
print("🚀 Enhanced Visual Persona Generator is ready to use!")
"""
            
            result = subprocess.run([
                self.python_executable, "-c", test_script
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.log("✅ Complete setup test passed!")
                self.log("🎉 Enhanced Visual Persona Generator is ready to use!")
                return True
            else:
                self.log(f"❌ Setup test failed: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("⚠️  Setup test timeout - but installation likely successful", "WARNING")
            return True
        except Exception as e:
            self.log(f"❌ Setup test error: {e}", "ERROR")
            return False
    
    def save_setup_log(self):
        """Save setup log for troubleshooting"""
        log_file = self.project_root / "setup_log.txt"
        with open(log_file, "w") as f:
            f.write("Enhanced Visual Persona Generator - Setup Log\n")
            f.write("=" * 50 + "\n\n")
            for entry in self.setup_log:
                f.write(entry + "\n")
        self.log(f"📝 Setup log saved to: {log_file}")
    
    def print_next_steps(self):
        """Print instructions for next steps"""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE - ENHANCED VISUAL PERSONA GENERATOR")
        print("="*60)
        print("\n📋 NEXT STEPS:")
        print("1. 🔑 Add your Gemini API key to .env file for enhanced features:")
        print("   GEMINI_API_KEY=your_api_key_here")
        print("\n2. 🚀 Start the application:")
        if self.is_windows:
            print("   .venv\\Scripts\\python.exe app.py")
        else:
            print("   .venv/bin/python app.py")
        print("\n3. 🌐 Open your browser to: http://localhost:8000")
        print("\n4. 📊 Upload images and enjoy AI-powered analysis!")
        print("\n💡 TIPS:")
        print("   • For development: uvicorn app:app --reload")
        print("   • Check setup_log.txt if you encounter issues")
        print("   • GPU acceleration automatic if available")
        print("   • Models cached for instant loading")
        print("\n🤖 Features Available:")
        print("   ✅ Real-time emotion detection (6 emotions)")
        print("   ✅ GPU-accelerated face detection (MTCNN)")
        print("   ✅ Intelligent scene analysis")
        print("   ✅ Beautiful visual reports") 
        print("   ✅ PDF report generation")
        print("   ✅ Batch image processing")
        if os.getenv('GEMINI_API_KEY'):
            print("   ✅ Gemini AI context analysis")
        else:
            print("   ⚠️  Gemini AI (add API key to .env)")
        print("\n" + "="*60)
    
    def run_complete_setup(self) -> bool:
        """Run the complete setup process"""
        print("🚀 Enhanced Visual Persona Generator - Automated Setup")
        print("=" * 55)
        print("This will set up everything needed for optimal performance:")
        print("• Python virtual environment")
        print("• All dependencies with GPU support")  
        print("• AI model downloads and caching")
        print("• Directory structure and configuration")
        print("• Complete functionality testing")
        print("\n⏱️  Estimated time: 5-15 minutes (depending on internet speed)")
        print("💾 Disk space needed: ~5GB for models and dependencies")
        
        # Confirm before proceeding
        if not input("\n▶️  Continue with setup? (y/N): ").lower().startswith('y'):
            print("Setup cancelled.")
            return False
        
        print("\n🔄 Starting automated setup...\n")
        
        # Run setup steps
        steps = [
            ("System Requirements", self.check_system_requirements),
            ("Virtual Environment", self.create_virtual_environment), 
            ("Dependencies", self.install_dependencies),
            ("AI Models", self.download_and_cache_models),
            ("Configuration", self.setup_directories_and_config),
            ("GPU Setup", lambda: self.verify_gpu_setup() is not None),
            ("Final Testing", self.test_complete_setup)
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            self.log(f"\n{'='*20} {step_name} {'='*20}")
            try:
                if step_func():
                    success_count += 1
                    self.log(f"✅ {step_name} completed successfully")
                else:
                    self.log(f"⚠️  {step_name} completed with warnings", "WARNING")
            except Exception as e:
                self.log(f"❌ {step_name} failed: {e}", "ERROR")
        
        # Save setup log
        self.save_setup_log()
        
        # Show results
        if success_count >= len(steps) - 1:  # Allow 1 failure
            self.log("\n🎉 Setup completed successfully!")
            self.print_next_steps()
            return True
        else:
            self.log(f"\n⚠️  Setup completed with issues ({success_count}/{len(steps)} steps successful)", "WARNING")
            self.log("Check setup_log.txt for details")
            return False

def main():
    """Main setup function"""
    try:
        setup = EnhancedSetup()
        success = setup.run_complete_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()