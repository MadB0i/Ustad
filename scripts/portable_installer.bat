@echo off
setlocal enabledelayedexpansion

echo.
echo ===============================================
echo    Ustad - Local LLM Distillation Studio
echo              Portable Installer
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Python 3.10-3.12 is required but not found.
    echo.
    echo Please download and install Python from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Found Python %PYTHON_VERSION%

REM Create portable directory
set "INSTALL_DIR=D:\Ustad-Portable"
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
cd /d "%INSTALL_DIR%"

echo [INFO] Installing to: %INSTALL_DIR%

REM Create virtual environment
echo [INFO] Creating virtual environment...
python -m venv venv
if !errorlevel! neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install PyTorch
echo [INFO] Installing PyTorch (CUDA 12.6 - may take 5-10 minutes)...
pip install torch --index-url https://download.pytorch.org/whl/cu126 --quiet --no-warn-script-location

REM Install other dependencies
echo [INFO] Installing AI/ML dependencies...
pip install transformers>=4.45 peft>=0.13 accelerate>=1.1 bitsandbytes>=0.43 safetensors>=0.4 --quiet --no-warn-script-location

echo [INFO] Installing web server dependencies...
pip install fastapi>=0.115 "uvicorn[standard]>=0.30" httpx>=0.27 --quiet --no-warn-script-location

echo [INFO] Installing Hugging Face Hub...
pip install "huggingface_hub>=0.25" --quiet --no-warn-script-location

REM Download Ustad files from GitHub
echo [INFO] Downloading Ustad application files...
curl -L -o ustad.zip https://github.com/MadB0i/Ustad/archive/refs/heads/main.zip
if !errorlevel! neq 0 (
    echo [ERROR] Failed to download Ustad files
    pause
    exit /b 1
)

REM Extract files
powershell -command "Expand-Archive -Path 'ustad.zip' -DestinationPath '.' -Force"
move Ustad-main\* .
rmdir /s /q Ustad-main
del ustad.zip

REM Create startup script
echo @echo off > start_ustad.bat
echo cd /d "%INSTALL_DIR%" >> start_ustad.bat
echo call venv\Scripts\activate.bat >> start_ustad.bat
echo echo Starting Ustad... >> start_ustad.bat
echo echo Web interface will open at http://127.0.0.1:8177 >> start_ustad.bat
echo python -m backend.server >> start_ustad.bat
echo pause >> start_ustad.bat

REM Create desktop shortcut script
echo @echo off > create_shortcut.bat
echo set "TARGET=%INSTALL_DIR%\start_ustad.bat" >> create_shortcut.bat
echo set "SHORTCUT=%%USERPROFILE%%\Desktop\Ustad.lnk" >> create_shortcut.bat
echo powershell -command "^$WshShell = New-Object -comObject WScript.Shell; ^$Shortcut = ^$WshShell.CreateShortcut('%%SHORTCUT%%'); ^$Shortcut.TargetPath = '%%TARGET%%'; ^$Shortcut.WorkingDirectory = '%INSTALL_DIR%'; ^$Shortcut.Save()" >> create_shortcut.bat

echo.
echo ===============================================
echo           Installation Complete!
echo ===============================================
echo.
echo Installation Location: %INSTALL_DIR%
echo.
echo To start Ustad:
echo   1. Double-click: start_ustad.bat
echo   2. Or run: create_shortcut.bat (creates desktop shortcut)
echo.
echo The web interface will open at:
echo   http://127.0.0.1:8177
echo.
echo Note: Make sure Ollama is installed and running
echo   Download from: https://ollama.ai
echo.
pause