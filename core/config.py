"""
core/config.py
──────────────
Centralized configuration loading and validation.
"""
import yaml
from pathlib import Path
from typing import Optional


def load_config(path: Optional[str] = None) -> dict:
    """Load and validate the training configuration YAML."""
    config_path = Path(path or "configs/training_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required = ["model", "data", "lora", "training"]
    for section in required:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    return config
