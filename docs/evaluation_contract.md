# Evaluation Contract

Success is defined by the following "Top 0.1%" performance targets:

| Metric | Target | Description |
| :--- | :--- | :--- |
| **OCA** | > 70% | Operation Classification Accuracy |
| **tIoU@0.5** | > 0.50 | Temporal Intersection over Union (Localization) |
| **AA@1** | > 50% | Anticipation Accuracy (The "Next Step" Prediction) |
| **Grammar Score** | < 0.20 | Frobenius distance from Ground Truth Grammar Matrix |

## Reproducibility Rules
- Use exactly 30 clips from **Subject U0108**.
- Clips must be selected **alphabetically by clip_id**.
- Final report must compare **Base Model** vs. **Fine-tuned Model**.
