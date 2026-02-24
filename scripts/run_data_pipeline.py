#!/usr/bin/env python3
"""
scripts/run_data_pipeline.py
─────────────────────────────
Convenience script: runs the full data pipeline steps in sequence.
Usage: python scripts/run_data_pipeline.py [--mock]
"""
import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Generate mock data first")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    # Step 1: Generate mock data if requested
    if args.mock:
        print("=== Step 1: Generating mock dataset ===")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_mock_data.py")],
            check=True, cwd=str(ROOT)
        )

    # Step 2: Run data pipeline
    print(f"=== Step 2: Running data pipeline (split={args.split}) ===")
    cmd = [
        sys.executable, str(ROOT / "data_pipeline.py"),
        "--config", str(ROOT / "configs" / "training_config.yaml"),
        "--split", args.split,
    ]
    if args.max_clips:
        cmd += ["--max-clips", str(args.max_clips)]
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    print("=== Data pipeline complete ===")

if __name__ == "__main__":
    main()
