# End-to-End Setup Guide: From Scratch to Production

This guide covers the complete procedure to download the OpenPack dataset, prepare the VLM pipeline, train the model, and deploy the inference API.

---

## 1. Prerequisites

- **OS**: Windows (with WSL2 Recommended) or Linux.
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 16GB+ VRAM (for training/inference). 
  - *Note: You can run the data pipeline and mock verification on a CPU/Mac.*
- **Tools**: `git`, `curl`, `conda` or `python -m venv`.

---

## 2. Dataset Acquisition (OpenPack)

The OpenPack dataset is hosted at [open-pack.org](https://open-pack.org/).

1. **Register**: Sign up at the official site to get access.
2. **Download Key Components**:
   - `Kinect RGB (Front)`: The primary video stream (~53 hours).
   - `Annotations`: Operation-level JSON/CSV files.
3. **Directory Structure**:
   Extract the dataset so it follows this pattern:
   ```bash
   data/openpack/
   ├── U0101/
   │   ├── S0100/
   │   │   ├── operation/annotation.json
   │   │   └── video/kinect_front.mp4
   ...
   ```
4. **Update Config**:
   Open `configs/training_config.yaml` and set `data.openpack_root` to your path.

---

## 3. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd vlm-temporal-logistics

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Local Verification (Mock Mode)

If you don't have the full dataset yet, you can verify every line of code using the **Mock Data Suite**:

```bash
# 1. Generate artificial video/annotation data
python scripts/generate_mock_data.py

# 2. Run the pipeline on mock data
# This generates .tar shards and the Procedural Grammar Matrix
python scripts/run_data_pipeline.py --max-clips 20 --split train
```

---

## 5. Training (Fine-Tuning)

### Option A: Local GPU
```bash
python scripts/run_training.py
```

### Option B: Cloud (Recommended for Free Tier)
1. Upload `finetune.ipynb` to [Kaggle](https://www.kaggle.com/).
2. Attach the **OpenPack** dataset (available in Kaggle Datasets).
3. Run all cells. It will automatically save checkpoints to `/kaggle/working/checkpoints`.

---

## 6. Evaluation

Compare the base model against your fine-tuned adapter:

```bash
python scripts/run_evaluation.py
```
Check `results.json` for **OCA**, **tIoU**, and **AA@1** scores.

---

## 7. Deployment

### Run the FastAPI Server
```bash
python scripts/run_api.py
```

### Test with a Video
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/my_video.mp4"
```

### Docker Deployment
```bash
docker-compose up --build vlm-api
```

---

## 8. Failure Recovery

- **OOM**: See `docs/failure_modes.md` to reduce batch size or frames.
- **Session Timeout**: The training is crash-tolerant. Just restart `run_training.py` and it will resume from the last 50-step checkpoint.
