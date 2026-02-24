#!/usr/bin/env python3
"""
scripts/run_training.py
────────────────────────
Convenience script: starts or resumes LoRA fine-tuning.
Usage: python scripts/run_training.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

def main():
    finetune_script = ROOT / "training" / "finetune.py"
    config = ROOT / "configs" / "training_config.yaml"

    if not finetune_script.exists():
        print(f"ERROR: {finetune_script} not found")
        sys.exit(1)

    print("=== Starting/Resuming LoRA Fine-Tuning ===")
    subprocess.run(
        [sys.executable, str(finetune_script), "--config", str(config)],
        check=True, cwd=str(ROOT)
    )
    print("=== Training complete ===")

if __name__ == "__main__":
    main()
