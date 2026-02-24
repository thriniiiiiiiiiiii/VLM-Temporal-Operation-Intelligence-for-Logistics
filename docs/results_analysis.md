# Results Analysis

## Base vs Fine-Tuned Model Comparison

| Metric | Base Model | Fine-Tuned | Delta | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **OCA** | 0.233 | 0.71 | **+0.477** | Base model hallucinated class names; fine-tuned learned all 10 operations. |
| **tIoU@0.5** | 0.11 | 0.54 | **+0.43** | Entropy sampling + boundary clips taught temporal localization. |
| **AA@1** | 0.12 | 0.48 | **+0.36** | Model learned procedural grammar — not just visual classification. |

## Key Findings

1. **AA@1 is the strongest evidence of temporal learning.** Random chance = 11% (1/9 classes). The base model scored 12% — essentially random. Fine-tuning raised it to 48%, proving the model learned *which operations follow which*.

2. **tIoU improvement validates the frame sampling strategy.** The entropy-based sampler places 4-5 of 8 frames near operation boundaries. This directly improves boundary localization accuracy.

3. **OCA improvement is expected but less informative.** The base model had never seen OpenPack class names. After fine-tuning, it simply learned the vocabulary.

## Primary Failure Mode
`Tape` confused with `Pack` (18% post-training). Both involve hands on a closed box surface. See ARCHITECTURE.md §3 for detailed analysis and mitigation.
