# Temporal Learning Theory

## How Temporal Reasoning Emerges from Fine-Tuning

The base VLM understands individual images. After fine-tuning on boundary-aware clips, it learns to **compare frames across time**, recognizing that visual changes between frame 1 and frame 8 signal operation transitions.

### Key Mechanisms

1. **Entropy-Sampled Frame Sequences**: Dense sampling near boundaries forces the model to attend to transition-specific visual cues (tape dispenser entering/leaving, hand repositioning).

2. **Procedural Grammar via Supervision**: By repeatedly predicting `anticipated_next_operation`, the model internalizes transition probabilities (e.g., `Tape → Put Items` at 95%). This is grammar learning without explicit rule coding.

3. **Temporal Grounding via Segment Labels**: The `temporal_segment` target teaches start/end frame prediction, requiring the model to localize *when* an operation occurs within the clip — not just *what* it is.

### Why Boundary Clips Are Essential
Without boundary clips, the model only sees stable mid-operation footage. It learns classification but **never** learns what transitions look like. Boundary clips are the critical training signal for tIoU and AA@1 metrics.

### Why Anticipation Matters
A system that classifies the present is reactive. A system that predicts the next step is **proactive**. In logistics, a 3-5 second prediction window enables pre-staging of downstream equipment and detection of procedural deviations.
