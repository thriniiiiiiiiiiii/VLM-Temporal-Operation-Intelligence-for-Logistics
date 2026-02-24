#!/usr/bin/env bash
# scripts/setup_env.sh
# ─────────────────────
# One-shot environment setup. Run once after cloning the repo.

set -euo pipefail
echo "==> Setting up VLM Temporal Operations environment"

# 1. Python virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch with CUDA 12.1
pip install --upgrade pip
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining deps
pip install -r requirements.txt

# 4. Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

echo "==> Setup complete. Activate with: source .venv/bin/activate"
