@echo off
REM Setup conda environment for the ultrakillLLM bot on Windows

set "ENV_NAME=ultrakill"

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Conda not found. Install Anaconda or Miniconda first.
    exit /b 1
)

REM Create environment if it does not already exist
conda env list | findstr /b /c:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    conda create -y -n "%ENV_NAME%" python=3.11
)

call conda activate "%ENV_NAME%"

REM Install core packages
conda install -y numpy gymnasium mss requests

REM Install remaining packages via pip
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3 opencv-python-headless pytesseract dxcam pywin32 pytest

echo Environment '%ENV_NAME%' is ready and active.
