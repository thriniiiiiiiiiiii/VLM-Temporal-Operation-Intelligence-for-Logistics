#!/usr/bin/env python3
"""
scripts/generate_mock_data.py
─────────────────────────────
Generates WebDataset shards from training_data_samples/.
File naming matches what OpenPackDataset._index_shards() expects:
  {clip_id}.meta.json    — metadata
  {clip_id}.frame_00.jpg — frame 0
  ...
  {clip_id}.target.json  — ground truth label

Usage:
    python scripts/generate_mock_data.py [--repo-root /kaggle/working/repo] [--split train]
"""

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np

REPO_ROOT_DEFAULT = Path(__file__).parent.parent

OPERATIONS = [
    "Box Setup", "Inner Packing", "Tape", "Put Items",
    "Pack", "Wrap", "Label", "Final Check", "Idle",
]


def make_shards(repo_root: Path, split: str = "train"):
    samples_dir = repo_root / "training_data_samples"
    shard_dir   = Path("/kaggle/working/shards") / split
    shard_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(samples_dir.glob("sample_*"))
    if not sample_dirs:
        print(f"[WARN] No samples found in {samples_dir} — generating synthetic data")
        _generate_synthetic_shards(shard_dir, n=20)
        return

    print(f"[INFO] Found {len(sample_dirs)} samples in {samples_dir}")

    shard_path = shard_dir / "shard-00000.tar"
    with tarfile.open(shard_path, "w") as tf:
        for idx, sample_dir in enumerate(sample_dirs):
            meta_path = sample_dir / "metadata.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            clip_id    = meta.get("clip_id", f"clip_{idx:04d}")
            dominant   = meta.get("dominant_operation", OPERATIONS[idx % len(OPERATIONS)])
            next_op    = meta.get("anticipated_next_operation", OPERATIONS[(idx + 1) % len(OPERATIONS)])
            total_fr   = meta.get("total_frames", 128)

            # --- meta.json ---
            meta_bytes = json.dumps({"clip_id": clip_id}).encode()
            info = tarfile.TarInfo(name=f"{clip_id}.meta.json")
            info.size = len(meta_bytes)
            tf.addfile(info, io.BytesIO(meta_bytes))

            # --- frame_XX.jpg files ---
            frames = sorted(sample_dir.glob("frame_*.jpg"))
            for fi, fp in enumerate(frames):
                data = fp.read_bytes()
                info = tarfile.TarInfo(name=f"{clip_id}.frame_{fi:02d}.jpg")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

            # --- target.json ---
            target = {
                "dominant_operation":          dominant,
                "anticipated_next_operation":  next_op,
                "temporal_segment": {
                    "start_frame": int(total_fr * 0.3),
                    "end_frame":   int(total_fr * 0.7),
                },
            }
            target_bytes = json.dumps(target).encode()
            info = tarfile.TarInfo(name=f"{clip_id}.target.json")
            info.size = len(target_bytes)
            tf.addfile(info, io.BytesIO(target_bytes))

    size_kb = shard_path.stat().st_size // 1024
    print(f"[OK] Shard written: {shard_path}  ({size_kb} KB, {len(sample_dirs)} clips)")


def _generate_synthetic_shards(shard_dir: Path, n: int = 20):
    """Fallback: generate fully synthetic data when no samples exist."""
    from PIL import Image

    shard_path = shard_dir / "shard-00000.tar"
    with tarfile.open(shard_path, "w") as tf:
        for idx in range(n):
            clip_id  = f"synthetic_{idx:04d}"
            dominant = OPERATIONS[idx % len(OPERATIONS)]
            next_op  = OPERATIONS[(idx + 1) % len(OPERATIONS)]
            total_fr = 128

            # meta.json
            meta_bytes = json.dumps({"clip_id": clip_id}).encode()
            info = tarfile.TarInfo(name=f"{clip_id}.meta.json")
            info.size = len(meta_bytes)
            tf.addfile(info, io.BytesIO(meta_bytes))

            # 8 frame JPEGs
            for fi in range(8):
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                data = buf.getvalue()
                info = tarfile.TarInfo(name=f"{clip_id}.frame_{fi:02d}.jpg")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

            # target.json
            target = {
                "dominant_operation":         dominant,
                "anticipated_next_operation": next_op,
                "temporal_segment": {
                    "start_frame": int(total_fr * 0.3),
                    "end_frame":   int(total_fr * 0.7),
                },
            }
            target_bytes = json.dumps(target).encode()
            info = tarfile.TarInfo(name=f"{clip_id}.target.json")
            info.size = len(target_bytes)
            tf.addfile(info, io.BytesIO(target_bytes))

    print(f"[OK] Synthetic shard written: {shard_path}  ({n} clips)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    make_shards(Path(args.repo_root), split=args.split)
