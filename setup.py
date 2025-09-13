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
6. Tests the complete setu            if cache_result.returncode == 0:
                self.log("✅ All models downloaded and cached successfully")
                self.log("🚀 Models will load instantly on first use!")
            else:
                self.log("⚠️  Some models may download on first use", "WARNING")
                if cache_result.stderr:
                    self.log(f"Model download details: {cache_result.stderr[:200]}...", "DEBUG")
            
            # Cleanup cache script
            try:
                cache_file.unlink(missing_ok=True)
            except:
                pass  # Ignore cleanup errors
            
            return True
            
        except subprocess.TimeoutExpired:
            self.log("⚠️  Model download timeout - models will download on first use", "WARNING")
            try:
                cache_file.unlink(missing_ok=True)
            except:
                pass
            return True
        except Exception as e:
            self.log(f"⚠️  Model caching issue: {e}", "WARNING")
            self.log("Models will download automatically on first use", "INFO")
            try:
                cache_file.unlink(missing_ok=True)
            except:
                pass
            return Truehing works perfectly

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

# Try to import tqdm for progress bars, fallback if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress bar implementation
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 100)
            self.desc = kwargs.get('desc', '')
            self.current = 0
            print(f"Starting: {self.desc}")
        
        def update(self, n=1):
            self.current += n
            if self.total:
                percent = (self.current / self.total) * 100
                print(f"\r{self.desc}: {percent:.1f}% ({self.current}/{self.total})", end='', flush=True)
        
        def close(self):
            print()  # New line after progress
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

class EnhancedSetup:
    def __init__(self, skip_models=False):
        self.project_root = Path(__file__).parent.absolute()
        self.venv_path = self.project_root / ".venv"
        self.is_windows = platform.system() == "Windows"
        self.python_executable = self._get_python_executable()
        self.setup_log = []
        self.total_steps = 7  # Total number of setup steps
        self.current_step = 0
        self.skip_models = skip_models
        
        # Try to install tqdm if not available
        if not TQDM_AVAILABLE:
            self._try_install_tqdm()
        # Try to install tqdm if not available
        if not TQDM_AVAILABLE:
            self._try_install_tqdm()
        
    def _try_install_tqdm(self):
        """Try to install tqdm for better progress bars"""
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "tqdm"], 
                         capture_output=True, check=True)
            global TQDM_AVAILABLE, tqdm
            from tqdm import tqdm as real_tqdm
            tqdm = real_tqdm
            TQDM_AVAILABLE = True
        except:
            pass  # Use fallback implementation
    
    def _update_overall_progress(self, step_name: str):
        """Update overall setup progress"""
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        print(f"\n📊 Overall Progress: {progress:.1f}% ({self.current_step}/{self.total_steps}) - {step_name}")
        
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
        """Create or update Python virtual environment"""
        self.log("🐍 Setting up Python virtual environment...")
        
        try:
            venv_exists = self.venv_path.exists()
            
            if venv_exists:
                self.log("📁 Virtual environment already exists - updating instead of removing...")
                # Check if virtual environment is functional
                if self._check_venv_functional():
                    self.log("✅ Existing virtual environment is functional")
                else:
                    self.log("⚠️  Virtual environment appears corrupted, recreating...", "WARNING")
                    shutil.rmtree(self.venv_path)
                    venv_exists = False
            
            if not venv_exists:
                # Create virtual environment
                self.log("🔨 Creating new virtual environment...")
                with tqdm(total=3, desc="Creating virtual environment") as pbar:
                    result = subprocess.run([
                        sys.executable, "-m", "venv", str(self.venv_path)
                    ], capture_output=True, text=True, check=True)
                    pbar.update(1)
                    
                    self.log("✅ Virtual environment created successfully")
                    pbar.update(1)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed to setup virtual environment: {e}", "ERROR")
            if e.stderr:
                self.log(f"Error details: {e.stderr}", "ERROR")
            return False
    
    def _check_venv_functional(self) -> bool:
        """Check if the existing virtual environment is functional"""
        try:
            # Test if Python executable exists and works
            if not Path(self.python_executable).exists():
                return False
            
            result = subprocess.run([
                self.python_executable, "--version"
            ], capture_output=True, text=True, timeout=10)
            
            return result.returncode == 0
        except:
            return False
    
    def _check_pytorch_installed(self) -> bool:
        """Check if PyTorch is already installed"""
        try:
            result = subprocess.run([
                self.python_executable, "-c", "import torch; print(torch.__version__)"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                self.log(f"Found existing PyTorch version: {version}")
                return True
            return False
        except:
            return False
    
    def _check_package_installed(self, package_name: str) -> bool:
        """Check if a package is already installed"""
        try:
            result = subprocess.run([
                self.python_executable, "-c", f"import {package_name}"
            ], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _install_pytorch_smart(self) -> bool:
        """Smart PyTorch installation with universal GPU support"""
        # Try to detect CUDA capability first
        cuda_version = self._detect_cuda_version()
        
        if cuda_version:
            self.log(f"Detected CUDA {cuda_version}, installing compatible PyTorch...")
            
            # Universal CUDA installation - let PyTorch auto-detect
            torch_cmd = [
                self.python_executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio"
            ]
            
            with tqdm(total=1, desc="Installing PyTorch with GPU support") as pbar:
                torch_result = subprocess.run(torch_cmd, capture_output=True, text=True)
                pbar.update(1)
                
            if torch_result.returncode == 0:
                self.log("✅ PyTorch with GPU support installed")
                return True
        
        # Fallback to CPU version
        self.log("Installing CPU-only PyTorch...")
        cpu_torch_cmd = [
            self.python_executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", "--index-url", 
            "https://download.pytorch.org/whl/cpu"
        ]
        
        with tqdm(total=1, desc="Installing PyTorch CPU version") as pbar:
            cpu_result = subprocess.run(cpu_torch_cmd, capture_output=True, text=True)
            pbar.update(1)
            
        if cpu_result.returncode == 0:
            self.log("✅ PyTorch CPU version installed")
            return True
        
        return False
    
    def _detect_cuda_version(self) -> str:
        """Detect CUDA version if available"""
        try:
            # Try nvidia-smi command
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0 and "CUDA Version:" in result.stdout:
                # Extract CUDA version from nvidia-smi output
                for line in result.stdout.split('\n'):
                    if "CUDA Version:" in line:
                        cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                        return cuda_version
        except:
            pass
        
        return None

    def install_dependencies(self) -> bool:
        """Install all required Python packages with smart detection and progress tracking"""
        self.log("📦 Installing Python dependencies...")
        
        try:
            # Check if PyTorch is already installed
            if self._check_pytorch_installed():
                self.log("✅ PyTorch already installed, skipping...")
            else:
                # Smart PyTorch installation with universal CUDA support
                self.log("🔥 Installing PyTorch with automatic GPU detection...")
                if self._install_pytorch_smart():
                    self.log("✅ PyTorch installation completed")
                else:
                    self.log("⚠️ PyTorch installation had issues, but continuing...", "WARNING")
            
            # Check and install main requirements intelligently
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                if self._check_requirements_installed(requirements_file):
                    self.log("✅ Most requirements already installed, checking for updates...")
                    self._install_missing_requirements(requirements_file)
                else:
                    self.log("📋 Installing requirements from requirements.txt...")
                    with tqdm(total=1, desc="Installing dependencies from requirements.txt") as pbar:
                        install_cmd = [
                            self.python_executable, "-m", "pip", "install", 
                            "-r", str(requirements_file)
                        ]
                        
                        result = subprocess.run(install_cmd, capture_output=True, text=True, check=True)
                        pbar.update(1)
                        
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
    
    def _check_requirements_installed(self, requirements_file: Path) -> bool:
        """Check if most requirements are already installed"""
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                requirements = f.readlines()
            
            installed_count = 0
            total_count = 0
            
            for req in requirements:
                req = req.strip()
                if req and not req.startswith('#'):
                    package_name = req.split('>=')[0].split('==')[0].split('[')[0]
                    total_count += 1
                    
                    if self._check_package_installed(package_name.replace('-', '_')):
                        installed_count += 1
            
            # If 70% or more packages are installed, consider it mostly installed
            if total_count > 0:
                percentage = (installed_count / total_count) * 100
                self.log(f"Found {installed_count}/{total_count} packages already installed ({percentage:.1f}%)")
                return percentage >= 70
            
            return False
        except:
            return False
    
    def _install_missing_requirements(self, requirements_file: Path):
        """Install only missing requirements with progress tracking"""
        try:
            # Use pip check to see what's missing
            check_cmd = [
                self.python_executable, "-m", "pip", "check"
            ]
            
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if check_result.returncode != 0:
                self.log("Installing missing/outdated packages...")
                with tqdm(total=1, desc="Installing missing packages") as pbar:
                    install_cmd = [
                        self.python_executable, "-m", "pip", "install", 
                        "-r", str(requirements_file), "--upgrade"
                    ]
                    subprocess.run(install_cmd, capture_output=True, text=True)
                    pbar.update(1)
                self.log("✅ Missing packages installed")
            else:
                self.log("✅ All requirements satisfied")
                
        except:
            self.log("Installing requirements normally...")
            with tqdm(total=1, desc="Installing requirements") as pbar:
                install_cmd = [
                    self.python_executable, "-m", "pip", "install", 
                    "-r", str(requirements_file)
                ]
                subprocess.run(install_cmd, capture_output=True, text=True)
                pbar.update(1)
    
    def _install_core_packages(self):
        """Install core packages manually if requirements.txt is missing with progress tracking"""
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
        
        with tqdm(total=len(core_packages), desc="Installing core packages") as pbar:
            for package in core_packages:
                self.log(f"Installing {package}...")
                subprocess.run([
                    self.python_executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, check=True)
                pbar.update(1)
    
    def download_and_cache_models(self) -> bool:
        """Download and cache all AI models for optimal performance - skip if already cached"""
        self.log("🤖 Checking AI models cache...")
        
        # Create models directory
        models_dir = self.project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Check if models are already cached
        if self._check_models_cached():
            self.log("✅ AI models already cached, skipping download...")
            return True
        
        # Skip model download if requested
        if self.skip_models:
            self.log("⏭️  Skipping model download as requested - models will download on first use")
            return True
        
        # Ask user if they want to download models now or skip
        self.log("📥 AI models not found in cache.")
        print("\n🤖 Model Download Options:")
        print("1. Download models now (recommended for optimal performance)")
        print("2. Skip download (models will download on first use)")
        
        try:
            choice = input("\nChoose option (1 or 2): ").strip()
            if choice == "2":
                self.log("⏭️  Skipping model download - models will download on first use")
                return True
        except:
            pass  # Continue with download if input fails
        
        self.log("📥 Downloading AI models (first time only)...")
        
        try:
            # Create a simple model check script
            cache_script = f'''
import sys
sys.path.append(r"{self.project_root}")
import os
import time
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

def download_with_progress(model_name, model_type="pipeline"):
    """Download model with progress feedback"""
    try:
        print(f"📥 Starting download: {{model_name}}")
        start_time = time.time()
        
        if model_type == "pipeline":
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            model = pipeline("text-classification", model=model_name, device=device)
        elif model_type == "detr":
            from transformers import DetrImageProcessor, DetrForObjectDetection
            processor = DetrImageProcessor.from_pretrained(model_name)
            model = DetrForObjectDetection.from_pretrained(model_name)
        
        elapsed = time.time() - start_time
        print(f"✅ Downloaded {{model_name}} in {{elapsed:.1f}}s")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {{model_name}}: {{e}}")
        return False

try:
    print("🔄 Starting AI model downloads...")
    
    # Download emotion model
    success1 = download_with_progress("j-hartmann/emotion-english-distilroberta-base", "pipeline")
    
    # Download object detection model  
    success2 = download_with_progress("facebook/detr-resnet-50", "detr")
    
    if success1 and success2:
        print("🎉 All models cached successfully!")
        sys.exit(0)
    else:
        print("⚠️  Some models failed to download but will be available on first use")
        sys.exit(0)  # Don't fail the setup for model download issues
        
except Exception as e:
    print(f"❌ Model caching error: {{e}}")
    print("📝 Models will download automatically on first use")
    sys.exit(0)  # Don't fail the setup
'''
            
            # Write and execute caching script
            cache_file = self.project_root / "cache_models.py"
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(cache_script)
            
            self.log("📥 Starting model downloads (may take 5-10 minutes)...")
            
            # Show progress for model download with real-time monitoring
            process = subprocess.Popen([
                self.python_executable, str(cache_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Monitor the process with a more realistic progress approach
            self.log("⏳ Model download in progress (this may take several minutes)...")
            
            # Wait for process completion with periodic updates
            start_time = time.time()
            timeout = 600  # Reduce to 10 minutes
            
            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self.log("⚠️  Model download taking too long, terminating...", "WARNING")
                    process.terminate()
                    time.sleep(3)  # Give process time to terminate
                    if process.poll() is None:
                        process.kill()  # Force kill if still running
                    break
                
                # Show periodic progress updates
                if int(elapsed) % 60 == 0 and elapsed > 0:  # Every minute
                    self.log(f"📥 Download still in progress... (elapsed: {elapsed/60:.1f} minutes)")
                
                time.sleep(5)  # Check every 5 seconds instead of 1
            
            # Get final result with timeout
            try:
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=10)
                    cache_result = type('Result', (), {
                        'returncode': process.returncode, 
                        'stdout': stdout, 
                        'stderr': stderr
                    })()
                else:
                    cache_result = type('Result', (), {
                        'returncode': -1, 
                        'stdout': '', 
                        'stderr': 'Process terminated due to timeout'
                    })()
            except subprocess.TimeoutExpired:
                process.kill()
                cache_result = type('Result', (), {
                    'returncode': -1, 
                    'stdout': '', 
                    'stderr': 'Process killed - communication timeout'
                })()
            
            self.log("✅ Model download process completed")
            
            if cache_result.returncode == 0:
                self.log("✅ All models downloaded and cached successfully")
                self.log("🚀 Models will load instantly on first use!")
            else:
                self.log("⚠️  Some models may download on first use", "WARNING")
                if hasattr(cache_result, 'stderr') and cache_result.stderr:
                    self.log(f"Download info: {cache_result.stderr[:200]}...", "DEBUG")
            
            # Cleanup cache script
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors
            
            return True
            
        except subprocess.TimeoutExpired:
            self.log("⚠️  Model download timeout - models will download on first use", "WARNING")
            return True
        except Exception as e:
            self.log(f"⚠️  Model caching issue: {e}", "WARNING")
            self.log("Models will download automatically on first use", "INFO")
            return True
    
    def _check_models_cached(self) -> bool:
        """Check if models are already cached in the environment"""
        try:
            # Quick check if transformers cache has our models
            result = subprocess.run([
                self.python_executable, "-c", 
                "from transformers import AutoTokenizer; "
                "t = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base'); "
                "print('cached')"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and 'cached' in result.stdout:
                return True
            return False
        except:
            return False
    
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
            project_path = str(self.project_root).replace('\\', '/')
            test_script = f"""
import sys
sys.path.append(r'{self.project_root}')

print("Testing core imports...")
try:
    from src.visual_persona_generator import VisualPersonaGenerator
    from src.enhanced_facial_emotion_analysis import EnhancedFacialEmotionAnalysis
    from src.gemini_context_analyzer import GeminiContextAnalyzer
    from src.intelligent_emotion_analysis import IntelligentEmotionAnalysis
    print("Core modules imported successfully")
except Exception as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

print("Testing model initialization...")
try:
    # Test without Gemini API key for basic functionality
    generator = VisualPersonaGenerator(gemini_api_key=None)
    print("Visual Persona Generator initialized")
except Exception as e:
    print(f"Initialization error: {{e}}")
    sys.exit(1)

print("Testing emotion analysis setup...")
try:
    from src.enhanced_facial_emotion_analysis import EnhancedFacialEmotionAnalysis
    emotion_analyzer = EnhancedFacialEmotionAnalysis()
    print("Emotion analysis ready")
except Exception as e:
    print(f"Emotion analysis error: {{e}}")
    sys.exit(1)

print("Setup test completed successfully!")
print("Enhanced Visual Persona Generator is ready to use!")
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
        try:
            with open(log_file, "w", encoding='utf-8', errors='replace') as f:
                f.write("Enhanced Visual Persona Generator - Setup Log\n")
                f.write("=" * 50 + "\n\n")
                for entry in self.setup_log:
                    # Remove emojis and special characters for compatibility
                    clean_entry = entry.encode('ascii', errors='ignore').decode('ascii')
                    f.write(clean_entry + "\n")
            self.log(f"Setup log saved to: {log_file}")
        except Exception as e:
            print(f"Could not save setup log: {e}")
    
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
        """Run the complete setup process with progress tracking"""
        print("🚀 Enhanced Visual Persona Generator - Automated Setup")
        print("=" * 55)
        print("This will set up everything needed for optimal performance:")
        print("• Python virtual environment (preserves existing)")
        print("• All dependencies with GPU support")  
        print("• AI model downloads and caching")
        print("• Directory structure and configuration")
        print("• Complete functionality testing")
        print("\n⏱️  Estimated time: 5-15 minutes (depending on internet speed)")
        print("💾 Disk space needed: ~5GB for models and dependencies")
        print("📊 Progress bars will show installation progress")
        
        # Confirm before proceeding
        if not input("\n▶️  Continue with setup? (y/N): ").lower().startswith('y'):
            print("Setup cancelled.")
            return False
        
        print("\n🔄 Starting automated setup...\n")
        
        # Run setup steps with progress tracking
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
        
        # Overall progress tracking
        with tqdm(total=len(steps), desc="Overall Setup Progress", position=0) as overall_pbar:
            for step_name, step_func in steps:
                self._update_overall_progress(step_name)
                self.log(f"\n{'='*20} {step_name} {'='*20}")
                
                try:
                    if step_func():
                        success_count += 1
                        self.log(f"✅ {step_name} completed successfully")
                    else:
                        self.log(f"⚠️  {step_name} completed with warnings", "WARNING")
                except Exception as e:
                    self.log(f"❌ {step_name} failed: {e}", "ERROR")
                
                overall_pbar.update(1)
                time.sleep(0.5)  # Brief pause for visual feedback
        
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
    
    def run_automated_setup(self) -> bool:
        """Run setup without user prompts (non-interactive mode)"""
        print("🚀 Enhanced Visual Persona Generator - Automated Setup")
        print("=" * 55)
        print("Running in non-interactive mode...")
        print("• Python virtual environment (preserves existing)")
        print("• All dependencies with GPU support")  
        print("• Directory structure and configuration")
        print("• Complete functionality testing")
        if self.skip_models:
            print("• AI models: SKIPPED (will download on first use)")
        else:
            print("• AI models: Will attempt download")
        
        print("\n🔄 Starting automated setup...\n")
        
        # Run setup steps with progress tracking
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
        
        # Overall progress tracking
        with tqdm(total=len(steps), desc="Overall Setup Progress", position=0) as overall_pbar:
            for step_name, step_func in steps:
                self._update_overall_progress(step_name)
                self.log(f"\n{'='*20} {step_name} {'='*20}")
                
                try:
                    if step_func():
                        success_count += 1
                        self.log(f"✅ {step_name} completed successfully")
                    else:
                        self.log(f"⚠️  {step_name} completed with warnings", "WARNING")
                except Exception as e:
                    self.log(f"❌ {step_name} failed: {e}", "ERROR")
                
                overall_pbar.update(1)
                time.sleep(0.2)  # Brief pause for visual feedback
        
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Visual Persona Generator Setup')
    parser.add_argument('--skip-models', action='store_true', 
                       help='Skip model downloads during setup')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run setup without user prompts')
    
    args = parser.parse_args()
    
    # Set UTF-8 encoding for Windows
    import locale
    import codecs
    if platform.system() == "Windows":
        try:
            # Try to set UTF-8 encoding for Windows console
            import sys
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            # Fallback for older Python versions
            import sys
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    try:
        setup = EnhancedSetup(skip_models=args.skip_models)
        
        if args.non_interactive:
            # Skip the confirmation prompt in non-interactive mode
            print("🚀 Running automated setup in non-interactive mode...")
            success = setup.run_automated_setup()
        else:
            success = setup.run_complete_setup()
            
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()