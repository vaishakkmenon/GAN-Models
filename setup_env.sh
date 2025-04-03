#!/bin/bash

# Script to set up a Python virtual environment for GAN training with DDP & AMP on RTX 5090s

# 1. Create and activate virtual environment
echo "[INFO] Creating a virtual environment..."
python3 -m venv gan_env

echo "[INFO] Activating the virtual environment..."
source gan_env/bin/activate  # Use 'source gan_env/Scripts/activate' on Windows

# 2. Install necessary Python dependencies
echo "[INFO] Installing PyTorch and dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install numpy

# Optional: Install NCCL if needed
echo "[INFO] Installing NCCL..."
sudo apt-get install libnccl-dev

# 3. Check if CUDA is available and PyTorch works
echo "[INFO] Verifying CUDA availability in PyTorch..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# 4. Set up project directories
echo "[INFO] Creating required directories..."
mkdir -p checkpoints generated

# 5. Verify GPUs are detected
echo "[INFO] Verifying GPUs are detected by CUDA..."
python -c "import torch; print('Number of GPUs available:', torch.cuda.device_count())"

# 6. Additional PyTorch checks (optional)
# python -c "import torch; print(torch.cuda.get_device_name(0))"

# 7. Instructions to the user
echo "[INFO] Setup is complete!"
echo "[INFO] Virtual environment 'gan_env' is activated. To activate it in the future, use: source gan_env/bin/activate"
echo "[INFO] Now you can run your script with: python full_script.py"