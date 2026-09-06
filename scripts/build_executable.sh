#!/bin/bash
# Build standalone executable for Ustad (Linux/macOS)
#
# This script creates a single-file executable that doesn't require Python installation.
# The resulting executable will be in dist/Ustad
#
# Requirements:
#   - Python 3.10-3.12 with venv activated
#   - PyInstaller installed (pip install pyinstaller)
#
# Usage:
#   ./scripts/build_executable.sh

set -e

echo "🔨 Building Ustad standalone executable..."

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated. Activating .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "❌ Virtual environment not found. Run quickstart.sh first."
        exit 1
    fi
fi

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "📦 Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous builds
if [ -d "build" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf build
fi
if [ -d "dist" ]; then
    echo "🧹 Cleaning previous dist..."
    rm -rf dist
fi

# Build executable
echo "🚀 Running PyInstaller..."
pyinstaller Ustad.spec

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo "📦 Executable: dist/Ustad"

    exe_size=$(du -h dist/Ustad | cut -f1)
    echo "📊 Size: $exe_size"

    # Make executable
    chmod +x dist/Ustad

    echo ""
    echo "⚠️  Notes:"
    echo "   • First run will be slower due to extraction"
    echo "   • Ollama must still be installed separately"
    echo "   • GPU drivers must be installed for GPU support"
    echo "   • The executable is standalone but large (~2-4GB)"
else
    echo ""
    echo "❌ Build failed. Check errors above."
    exit 1
fi
