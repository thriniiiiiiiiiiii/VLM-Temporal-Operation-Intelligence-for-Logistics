# VLM Temporal Operation Intelligence for Logistics

**Production-grade Vision-Language Model system for temporal video understanding in warehouse operations**

This repository implements an end-to-end Vision-Language Model (VLM) pipeline for temporal operation understanding, procedural workflow modeling, and next-action anticipation in real-world logistics environments.

The system performs:
- **Operation recognition** from short video clips
- **Temporal boundary localization** (start/end frames)
- **Procedural grammar learning** (sequence modeling)
- **Next-operation anticipation**
- **Resource-constrained training**
- **Production-ready deployment**


---

## Problem Statement

Traditional computer vision systems operate on static frames and fail to understand sequences of actions in industrial workflows.
This project builds a temporal VLM system capable of:
- Understanding sequential operations
- Detecting temporal boundaries between tasks
- Learning procedural logic
- Predicting future actions
- Operating under free-tier compute constraints
- Deploying as a production API

---

## Core Capabilities

- **Temporal video understanding** (not frame-level classification)
- **Boundary-aware clip sampling** (Entropy-based)
- **Procedural sequence modeling**
- **Anticipation learning**
- **Memory-efficient fine-tuning** (QLoRA 4-bit)
- **GPU-efficient training** (Streaming WebDataset)
- **Production deployment** (FastAPI + Docker)

---

## System Architecture

### High-Level Architecture
```
Raw Video → Data Engine → Entropy Sampling → WebDataset Shards
                                                   ↓
                                          LoRA Fine-Tuning (QLoRA 4-bit)
                                                   ↓
                                      Trained Model → FastAPI /predict + /analyze
                                                   ↓
                                           Evaluation Engine → results.json
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions.

---

## Repository Structure

```
.
├── docker-compose.yml           # FastAPI deployment config
├── Dockerfile                   # Container definition
├── data_pipeline.py             # OpenPack data loader + frame sampling
├── training_data_samples/       # 20 example training pairs
├── finetune.ipynb               # Fine-tuning notebook (Kaggle/GCP)
├── evaluate.py                  # Evaluation script
├── results.json                 # Base vs fine-tuned metrics
├── ARCHITECTURE.md              # System design & technical decisions
├── AGENTS.md                    # AI development log
├── api/
│   └── main.py                  # FastAPI: /predict, /analyze
├── model/
│   └── vlm.py                   # VLM engine (load, quantize, infer)
├── configs/
│   └── training_config.yaml     # All hyperparameters
└── docs/                        # Comprehensive phase documentation
```

---

## Dataset: OpenPack

Real-world warehouse packaging operations dataset.
- **Modality:** Kinect RGB (Frontal)
- **Resolution:** 480×640 @ 25 FPS
- **Operation classes:** Box Setup, Inner Packing, Tape, Put Items, Pack, Wrap, Label, Final Check, Idle, Unknown.
- **Splits:** Training (U0101–U0106), Validation (U0107), Test (U0108).

---

## Installation & Quick Start

### 1. Requirements
- Python 3.10+
- CUDA-compatible GPU (T4/A100)
- Docker + NVIDIA Container Toolkit

### 2. Setup
```bash
git clone https://github.com/thriniiiiiiiiiiii/VLM-Temporal-Operation-Intelligence-for-Logistics.git
pip install -r requirements.txt
```

### 3. Data Pipeline
```bash
python data_pipeline.py --config configs/training_config.yaml --split train
```

---

## Model Training

Fine-tuning is optimized for **Qwen2.5-VL-3B-Instruct** using QLoRA.
Refer to [finetune.ipynb](finetune.ipynb) for the full training loop on Kaggle (T4) or GCP (A100).

---

## Evaluation

```bash
python evaluate.py --config configs/training_config.yaml
```

### Results Summary

| Metric | Base Model | Fine-Tuned (3B) | Delta |
| :--- | :--- | :--- | :--- |
| **OCA** | 0.42 | 0.40 | -0.02 |
| **tIoU@0.5** | 0.31 | 1.00 | +0.69 |
| **AA@1** | 0.35 | 0.20 | -0.15 |

---

## API Deployment

```bash
docker-compose up --build
```

### Endpoint: `POST /predict`
**Example Request:**
```bash
curl -X POST http://localhost:8000/predict -F "file=@test_clip.mp4"
```

**Output Schema:**
```json
{
  "clip_id": "U0108_S0500_t0035",
  "dominant_operation": "Tape",
  "temporal_segment": {
    "start_frame": 14,
    "end_frame": 98
  },
  "anticipated_next_operation": "Put Items",
  "confidence": 0.87
}
```

