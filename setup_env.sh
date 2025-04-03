#!/bin/bash

# Project root and paths
PROJECT_ROOT="/workspace/GAN-Models"
SCRIPT_PATH="$PROJECT_ROOT/model/GAN/full_script.py"
VENV_DIR="$PROJECT_ROOT/gan_env"
PYTHON_VERSION=3.10

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment at $VENV_DIR..."
    python$PYTHON_VERSION -m venv "$VENV_DIR"
fi

# Activate the virtual environment
echo "[INFO] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
echo "[INFO] Installing PyTorch (CUDA 12.1) and dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other useful packages
pip install numpy matplotlib tqdm

# Export environment variables for Distributed Data Parallel
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Launch training
echo "[INFO] Running full_script.py with Distributed Data Parallel on 4 GPUs..."
python "$SCRIPT_PATH"

# Done
echo "[INFO] Training completed successfully."