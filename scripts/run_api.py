#!/usr/bin/env python3
"""
scripts/run_api.py
───────────────────
Convenience script: starts the FastAPI inference server.
Usage: python scripts/run_api.py [--port 8000]
"""
import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"=== Starting VLM API on port {args.port} ===")
    cmd = [
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--workers", "1",
    ]
    if args.reload:
        cmd.append("--reload")

    subprocess.run(cmd, cwd=str(ROOT))

if __name__ == "__main__":
    main()
