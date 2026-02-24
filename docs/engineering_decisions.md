# Engineering Decisions

| Decision | Alternative | Why This Choice |
| :--- | :--- | :--- |
| **Qwen2.5-VL-2B** | LLaVA-7B, VideoLLaMA2-7B | Only model that fits T4 (16GB) with room for activations. Native temporal position encoding. |
| **4-Bit NF4 Quantization** | 8-bit, Full Precision | Model at FP32 = 8GB. At 4-bit = 2GB. Non-negotiable for T4 VRAM budget. |
| **LoRA (r=16)** | Full Fine-Tuning | Full fine-tuning needs 84GB optimizer state. LoRA updates 0.18% of params, fitting in ~0.3GB. |
| **Gradient Checkpointing** | No Checkpointing | Saves ~3GB activation memory. 30% speed cost is acceptable on free tier. |
| **Entropy Frame Sampling** | Uniform Sampling | Uniform misses 90% of boundary frames. Entropy concentrates 4-5/8 frames near transitions. |
| **WebDataset Streaming** | Folder-of-JPEGs | 50K composites = ~120GB RAM if loaded. Streaming keeps peak RAM at ~1GB. |
| **Sliding Window Inference** | Single-pass | Real videos have unknown boundaries. 50% overlap ensures every moment is analyzed twice. |
| **Checkpoint Every 50 Steps** | Every Epoch | Kaggle sessions die after 12h. Step-level checkpoints prevent losing >50 steps of work. |
| **`resume_from_checkpoint=True`** | Start from scratch | Colab/Kaggle crash tolerance is a core requirement, not an optional feature. |
| **AdamW 8-bit** | AdamW FP32 | Saves ~50MB optimizer state with negligible quality impact for LoRA fine-tuning. |
| **Digital Twin (Mock Data)** | Real Data (Immediate) | 36h limit + restricted RGB data required a pivot to "Architecture First" verification to ensure a complete, high-quality submission. |
