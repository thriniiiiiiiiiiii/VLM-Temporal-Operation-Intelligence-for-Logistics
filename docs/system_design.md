# System Design: Temporal VLM

## 1. Data Contract
The atomic unit of training is the **Temporal Clip**:
- **Inputs**: 8 frames sampled via Shannon Entropy (biased towards boundaries).
- **Core Prompt**: Instructional template asking for Operation, Segment, and Next-Operation.
- **Answer**: Structured JSON following the logic of warehouse transitions.

## 2. Model Architecture
- **Backbone**: Qwen2.5-VL-2B (for native temporal position encoding).
- **Quantization**: NF4 4-bit (via BitsAndBytes).
- **PEFT**: LoRA on `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

## 3. Data Flow
1. **Raw Video** -> **Boundary Detection** -> **Clip Extraction**.
2. **Clips** -> **Entropy Sampling** -> **Annotation Mapping**.
3. **Training Pairs** -> **WebDataset Sharding**.
4. **LoRA Fine-tuning** -> **Checkpointing (Crash-Tolerant)**.
5. **Evaluation** -> **OCA/tIoU/AA Metrics** -> **Results JSON**.

## 4. Crash Tolerance (Critical)
- **Checkpoints**: Saved every 50 steps to Kaggle/Colab persistent storage.
- **Resume Logic**: Auto-detects latest `checkpoint-*` and restores optimizer state.
