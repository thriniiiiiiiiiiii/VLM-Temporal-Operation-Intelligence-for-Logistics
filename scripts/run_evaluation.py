#!/usr/bin/env python3
"""
scripts/run_evaluation.py
──────────────────────────
Convenience script: runs full evaluation on test set.
Usage: python scripts/run_evaluation.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

def main():
    eval_script = ROOT / "evaluate.py"
    config = ROOT / "configs" / "training_config.yaml"

    print("=== Running Evaluation (Base + Fine-tuned) ===")
    subprocess.run(
        [
            sys.executable, str(eval_script),
            "--config", str(config),
            "--data-root", str(ROOT / "mock_data"),
            "--n-clips", "30",
        ],
        check=True, cwd=str(ROOT)
    )
    print("=== Evaluation complete -- see results.json ===")

if __name__ == "__main__":
    main()
