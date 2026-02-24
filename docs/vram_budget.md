# VRAM Budget Calculation

| Component | VRAM Usage (Est.) | Rationale |
| :--- | :--- | :--- |
| **Base Model** | 2.0 GB | Qwen2.5-VL-2B in 4-bit NF4 |
| **LoRA Adapters** | 0.3 GB | Rank=16, all linear layers |
| **Activations (GC)** | 0.5 GB | With gradient checkpointing |
| **Optimizer States** | 0.1 GB | AdamW 8-bit for LoRA parameters |
| **CUDA Overhead** | ~6.0 GB | Context + Framework + KV Cache |
| **Total (Observed)** | **~9-10 GB** | **Fits Kaggle/Colab T4 (16GB)** |

## Memory Saving Strategy
1. **4-bit NF4**: Mandatory for loading 2B model on T4.
2. **Gradient Checkpointing**: Reduces activation memory from >3GB to ~500MB.
3. **LoRA**: Updates <1% of parameters, keeping optimizer states minimal.
