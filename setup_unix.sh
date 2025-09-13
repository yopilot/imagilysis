#!/bin/bash
# Enhanced Visual Persona Generator - Linux/Mac Setup Launcher
# =============================================================
# This script launches the Python setup script for easy installation

echo ""
echo "================================================================"
echo " Enhanced Visual Persona Generator - Automated Setup (Unix/Mac)"
echo "================================================================"
echo ""
echo "This will automatically set up everything needed:"
echo " • Python virtual environment"
echo " • All dependencies with GPU acceleration"
echo " • AI model downloads and caching"
echo " • Complete configuration and testing"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 not found"
    echo ""
    echo "Please install Python 3.11+ from your package manager:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo "  macOS: brew install python3"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo ""
    exit 1
fi

# Display Python version
echo "🐍 Python version:"
python3 --version
echo ""

# Check Python version requirement
python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Python 3.11+ required"
    echo ""
    echo "Please update Python from your package manager or:"
    echo "https://www.python.org/downloads/"
    echo ""
    exit 1
fi

# Make sure we're in the script directory
cd "$(dirname "$0")" || exit 1

# Run the setup script
echo "🚀 Launching automated setup..."
echo ""
python3 setup.py

# Check if setup was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Setup encountered issues. Check setup_log.txt for details."
    echo ""
    exit 1
else
    echo ""
    echo "✅ Setup completed! You can now run the application."
    echo ""
    echo "Quick start commands:"
    echo "  1. Start server: .venv/bin/python app.py"
    echo "  2. Open browser: http://localhost:8000"
    echo ""
    echo "Press Enter to exit..."
    read -r
fi