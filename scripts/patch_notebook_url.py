"""
scripts/patch_notebook_url.py
─────────────────────────────
Patches finetune.ipynb Cell 1 with the real Kaggle notebook URL.
Run once after cloning: python scripts/patch_notebook_url.py
"""
import json
from pathlib import Path

KAGGLE_URL = "https://www.kaggle.com/code/thrinainiaroori/finetune"
NB_PATH = Path(__file__).parent.parent / "finetune.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Patch Cell 0 (first cell — Markdown header)
cell = nb["cells"][0]
new_source = []
for line in cell["source"]:
    if "REPLACE WITH YOUR KAGGLE" in line or "YOUR_USERNAME" in line:
        continue  # drop placeholder lines
    new_source.append(line)

# Insert the real URL right after the title line
insert_idx = 1  # after "# Qwen2.5-VL ..." title line
new_source.insert(insert_idx, f"\n**Live Kaggle Notebook (GPU T4 x2 — QLoRA Training):**\n")
new_source.insert(insert_idx + 1, f"{KAGGLE_URL}\n")
new_source.insert(insert_idx + 2, "\n")

cell["source"] = new_source

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[OK] finetune.ipynb patched with URL: {KAGGLE_URL}")
