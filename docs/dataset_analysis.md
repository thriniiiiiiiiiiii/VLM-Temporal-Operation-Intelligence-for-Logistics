# Dataset Analysis: OpenPack

## Overview
OpenPack is an industrial logistics dataset featuring packaging work.

## Operation Classes (10)
- `Box Setup`
- `Inner Packing`
- `Tape`
- `Put Items`
- `Pack`
- `Wrap`
- `Label`
- `Final Check`
- `Idle`
- `Unknown`

## Data Splits
- **Training**: U0101, U0102, U0103, U0104, U0105, U0106
- **Validation**: U0107
- **Test**: U0108

## Modalitites
- **Primary**: Kinect Frontal RGB (15fps/30fps).
- **Secondary (Ignored)**: IMU, Depth, Keypoints.

## Key Challenges
- **Class Imbalance**: "Put Items" is significantly more frequent than "Box Setup".
- **Temporal Ambiguity**: Boundaries between "Tape" and "Pack" are visually subtle.
- **Subject Variation**: Different workers have distinct motions for the same operation.
