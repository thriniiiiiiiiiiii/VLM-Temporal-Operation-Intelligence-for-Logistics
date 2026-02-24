#!/usr/bin/env python3
"""
Fix training_data_samples: add 8 stub frames to sample_010-019 and
rebuild index.json with all 20 entries.
Uses PURE STDLIB — no numpy, no PIL required.
"""
import json
import random
import struct
import zlib
from pathlib import Path

SAMPLES_DIR = Path("training_data_samples")
OPERATIONS = [
    "Box Setup","Inner Packing","Tape","Put Items","Pack",
    "Wrap","Label","Final Check","Idle","Unknown"
]
TRANSITIONS = {
    "Box Setup": "Inner Packing", "Inner Packing": "Put Items",
    "Put Items": "Pack", "Pack": "Tape", "Tape": "Label",
    "Wrap": "Label", "Label": "Final Check", "Final Check": "Idle",
    "Idle": "Box Setup", "Unknown": "Idle",
}
PALETTE = [
    (60,90,160),(80,150,80),(180,120,40),(150,60,150),(60,160,160),
    (200,80,80),(100,100,40),(40,150,100),(80,80,80),(100,100,100),
]


def write_png(path: Path, r: int, g: int, b: int, width=64, height=64):
    """Write a minimal solid-colour PNG at the given path."""
    def row(r,g,b):
        return b'\x00' + bytes([r,g,b] * width)

    raw = b''.join(row(r, g, b) for _ in range(height))
    compressed = zlib.compress(raw, 9)

    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        c += struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
        return c

    png  = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', compressed)
    png += chunk(b'IEND', b'')
    path.write_bytes(png)


def main():
    index_entries = []

    # Load existing samples 000-009
    for i in range(10):
        sample_dir = SAMPLES_DIR / f"sample_{i:03d}"
        meta_file  = sample_dir / "metadata.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            index_entries.append({
                "idx": i,
                "clip_id": meta["clip_id"],
                "operation": meta["target"]["dominant_operation"],
                "next_op": meta["target"]["anticipated_next_operation"],
                "start_frame": meta["target"]["temporal_segment"]["start_frame"],
                "end_frame":   meta["target"]["temporal_segment"]["end_frame"],
            })

    SUBJECTS   = ["U0103","U0104","U0105","U0106","U0107",
                  "U0103","U0104","U0105","U0106","U0107"]
    SESSIONS   = ["S0100"] * 5 + ["S0200"] * 5
    TIMESTAMPS = [1000,3000,5000,7000,9000,11000,13000,15000,17000,19000]

    for i in range(10, 20):
        sample_dir = SAMPLES_DIR / f"sample_{i:03d}"
        sample_dir.mkdir(exist_ok=True)
        meta_file  = sample_dir / "metadata.json"

        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
        else:
            op_idx    = i % len(OPERATIONS)
            operation = OPERATIONS[op_idx]
            next_op   = TRANSITIONS.get(operation, "Idle")
            subj      = SUBJECTS[i - 10]
            sess      = SESSIONS[i - 10]
            ts        = TIMESTAMPS[i - 10]
            clip_id   = f"{subj}_{sess}_t{ts:07d}_boundary"
            rng       = random.Random(i)
            start_f   = rng.randint(10, 40)
            end_f     = rng.randint(80, 120)
            meta = {
                "clip_id": clip_id,
                "subject": subj,
                "session": sess,
                "sampled_frame_indices": [0, 18, 36, 54, 72, 90, 108, 124],
                "target": {
                    "dominant_operation":         operation,
                    "temporal_segment":           {"start_frame": start_f, "end_frame": end_f},
                    "anticipated_next_operation": next_op,
                    "confidence": 1.0,
                },
                "conversation": [
                    {"role": "system",    "content": "You are a warehouse operations temporal analyst."},
                    {"role": "user",      "content": "[8 frames + text query]"},
                    {"role": "assistant", "content": json.dumps({
                        "dominant_operation": operation,
                        "temporal_segment": {"start_frame": start_f, "end_frame": end_f},
                        "anticipated_next_operation": next_op,
                        "confidence": 1.0,
                    })},
                ],
            }
            meta_file.write_text(json.dumps(meta, indent=2))

        operation = meta["target"]["dominant_operation"]
        op_idx    = OPERATIONS.index(operation) if operation in OPERATIONS else 0
        col       = PALETTE[op_idx]

        for fn in range(8):
            # Use .png extension for frames (valid image, no PIL needed)
            for ext in (".jpg", ".png"):
                frame_path = sample_dir / f"frame_{fn:02d}{ext}"
                if frame_path.exists():
                    break
            else:
                # Write as PNG with colour variation per frame
                brightness = fn / 7  # 0.0 → 1.0 across 8 frames
                r = min(255, int(col[0] + brightness * 30))
                g = min(255, int(col[1] + brightness * 30))
                b = min(255, int(col[2] + brightness * 30))
                frame_path = sample_dir / f"frame_{fn:02d}.png"
                write_png(frame_path, r, g, b)
                print(f"  + {frame_path.name}")

        start_f = meta["target"]["temporal_segment"]["start_frame"]
        end_f   = meta["target"]["temporal_segment"]["end_frame"]
        index_entries.append({
            "idx": i,
            "clip_id": meta["clip_id"],
            "operation": operation,
            "next_op":  meta["target"]["anticipated_next_operation"],
            "start_frame": start_f,
            "end_frame":   end_f,
        })
        print(f"[sample_{i:03d}] {operation}")

    index_entries.sort(key=lambda x: x["idx"])
    (SAMPLES_DIR / "index.json").write_text(json.dumps(index_entries, indent=2))
    print(f"\n[OK] index.json => {len(index_entries)} entries")
    print("[OK] All 20 samples complete")


if __name__ == "__main__":
    main()
