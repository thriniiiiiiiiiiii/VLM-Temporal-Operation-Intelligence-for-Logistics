# Problem Statement: Temporal Operation Intelligence for Logistics

## Vision
To replace manual human surveillance of warehouse packaging operations with an autonomous AI system capable of understanding not just individual frames, but the temporal "story" of work.

## Core Objectives
1.  **Temporal Action Recognition**: Identify 10 discrete packaging operations (Box Setup, Tape, etc.) from 5-second video clips.
2.  **Temporal Grounding**: Pinpoint the exact start and end frames (boundaries) of these operations.
3.  **Next-Step Anticipation**: Predict the subsequent operation in the workflow based on procedural grammar.

## Constraints
- **Hardware**: NVIDIA T4 GPU (Kaggle/Colab) with 16GB VRAM.
- **Compute Strategy**: 4-bit NF4 Quantization + QLoRA + Gradient Checkpointing.
- **Data**: OpenPack Kinect RGB video (Subject split: U0101-U0106 Train, U0107 Val, U0108 Test).

## Success Definition
Measured via OCA (Classification), tIoU@0.5 (Localization), and AA@1 (Anticipation).
