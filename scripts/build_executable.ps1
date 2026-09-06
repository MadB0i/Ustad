# Build standalone executable for Ustad (Windows)
#
# This script creates a single-file executable that doesn't require Python installation.
# The resulting executable will be in dist/Ustad.exe
#
# Requirements:
#   - Python 3.10-3.12 with venv activated
#   - PyInstaller installed (pip install pyinstaller)
#
# Usage:
#   .\scripts\build_executable.ps1

$ErrorActionPreference = "Stop"

Write-Host "🔨 Building Ustad standalone executable..." -ForegroundColor Cyan

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated. Activating .venv..." -ForegroundColor Yellow
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        & .venv\Scripts\Activate.ps1
    } else {
        Write-Host "❌ Virtual environment not found. Run quickstart.ps1 first." -ForegroundColor Red
        exit 1
    }
}

# Check if PyInstaller is installed
try {
    pyinstaller --version | Out-Null
} catch {
    Write-Host "📦 Installing PyInstaller..." -ForegroundColor Yellow
    pip install pyinstaller
}

# Clean previous builds
if (Test-Path "build") {
    Write-Host "🧹 Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build
}
if (Test-Path "dist") {
    Write-Host "🧹 Cleaning previous dist..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force dist
}

# Build executable
Write-Host "🚀 Running PyInstaller..." -ForegroundColor Cyan
pyinstaller Ustad.spec

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Build successful!" -ForegroundColor Green
    Write-Host "📦 Executable: dist\Ustad.exe" -ForegroundColor Green

    $exeSize = (Get-Item "dist\Ustad.exe").Length / 1MB
    Write-Host "📊 Size: $($exeSize.ToString('F2')) MB" -ForegroundColor Cyan

    Write-Host "`n⚠️  Notes:" -ForegroundColor Yellow
    Write-Host "   • First run will be slower due to extraction" -ForegroundColor Gray
    Write-Host "   • Ollama must still be installed separately" -ForegroundColor Gray
    Write-Host "   • GPU drivers must be installed for GPU support" -ForegroundColor Gray
    Write-Host "   • The executable is standalone but large (~2-4GB)" -ForegroundColor Gray
} else {
    Write-Host "`n❌ Build failed. Check errors above." -ForegroundColor Red
    exit 1
}
