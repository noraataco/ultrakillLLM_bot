#!/usr/bin/env bash

# Setup conda environment for the ultrakillLLM bot
# This script creates a conda environment with the required
# dependencies and activates it.

set -euo pipefail

ENV_NAME="ultrakill"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found. Install Anaconda or Miniconda first." >&2
    exit 1
fi

# Create environment if it does not already exist
if ! conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
    conda create -y -n "$ENV_NAME" python=3.11
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install core packages
conda install -y numpy gymnasium mss requests

# Install remaining packages via pip (PyPI)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3 opencv-python-headless pytesseract dxcam pywin32 pytest

echo "Environment '$ENV_NAME' is ready. Dropping into a shell..."
exec "$SHELL" -i
