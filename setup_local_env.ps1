# setup_local_env.ps1

Write-Host "==> Setting up VLM Temporal Operations environment" -ForegroundColor Cyan

# 1. Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "--> Creating virtual environment (.venv)..."
    python -m venv .venv
} else {
    Write-Host "--> Virtual environment .venv already exists."
}

# 2. Activate virtual environment
Write-Host "--> Activating virtual environment..."
& .venv\Scripts\Activate.ps1

# 3. Upgrade pip
Write-Host "--> Upgrading pip..."
python -m pip install --upgrade pip

# 4. Install PyTorch with CUDA 12.1
Write-Host "--> Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# 5. Install remaining dependencies
Write-Host "--> Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt

# 6. Verify installation
Write-Host "--> Verifying installation..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

Write-Host "==> Setup complete!" -ForegroundColor Green
