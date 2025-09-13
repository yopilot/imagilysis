#!/usr/bin/env python3
"""
Quick Setup Script for Enhanced Visual Persona Generator
=======================================================

This is a simplified setup script that skips model downloads
and focuses on getting the environment ready quickly.
"""

import subprocess
import sys
import os
from pathlib import Path

def quick_setup():
    """Quick setup without model downloads"""
    print("🚀 Quick Setup - Enhanced Visual Persona Generator")
    print("=" * 50)
    print("This will set up the basic environment without downloading models.")
    print("Models will download automatically on first use.")
    
    project_root = Path(__file__).parent.absolute()
    venv_path = project_root / ".venv"
    is_windows = os.name == 'nt'
    
    if is_windows:
        python_exe = str(venv_path / "Scripts" / "python.exe")
        pip_exe = str(venv_path / "Scripts" / "pip.exe")
    else:
        python_exe = str(venv_path / "bin" / "python")
        pip_exe = str(venv_path / "bin" / "pip")
    
    try:
        # 1. Create virtual environment if needed
        if not venv_path.exists():
            print("🐍 Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print("✅ Virtual environment created")
        else:
            print("✅ Virtual environment already exists")
        
        # Install requirements
        requirements_file = project_root / "requirements.txt"
        if requirements_file.exists():
            print("📋 Installing requirements...")
            subprocess.run([python_exe, "-m", "pip", "install", "-r", str(requirements_file)], 
                          check=True)
            print("✅ Requirements installed")
        else:
            print("⚠️  No requirements.txt found")
        
        # Create directories
        print("📁 Creating directories...")
        for dir_name in ["data", "data/uploads", "models", "utils"]:
            (project_root / dir_name).mkdir(exist_ok=True)
        print("✅ Directories created")
        
        # Create .env file if needed
        env_file = project_root / ".env"
        if not env_file.exists():
            print("📄 Creating .env file...")
            env_content = """# Enhanced Visual Persona Generator Configuration
GEMINI_API_KEY=your_gemini_api_key_here
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
USE_GPU=auto
"""
            with open(env_file, "w") as f:
                f.write(env_content)
            print("✅ .env file created")
        
        print("\n🎉 Quick setup completed!")
        print("💡 To start the application:")
        if is_windows:
            print("   .venv\\Scripts\\python.exe app.py")
        else:
            print("   .venv/bin/python app.py")
        print("📝 Models will download automatically on first use")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_setup()
    sys.exit(0 if success else 1)