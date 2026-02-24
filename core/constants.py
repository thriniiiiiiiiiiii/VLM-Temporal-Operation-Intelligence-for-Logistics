"""
core/constants.py
─────────────────
Shared constants used across the project.
"""

# 10 OpenPack operation classes
OPERATION_CLASSES = [
    "Box Setup",
    "Inner Packing",
    "Tape",
    "Put Items",
    "Pack",
    "Wrap",
    "Label",
    "Final Check",
    "Idle",
    "Unknown",
]

# Operation ID mapping
OP_TO_ID = {name: idx for idx, name in enumerate(OPERATION_CLASSES)}
ID_TO_OP = {idx: name for idx, name in enumerate(OPERATION_CLASSES)}

# Data splits
TRAIN_SUBJECTS = ["U0101", "U0102", "U0103", "U0104", "U0105", "U0106"]
VAL_SUBJECTS = ["U0107"]
TEST_SUBJECTS = ["U0108"]

# Random seeds for reproducibility
SEED = 42
