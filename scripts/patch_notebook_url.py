"""
scripts/patch_notebook_url.py
─────────────────────────────
Patches finetune.ipynb with:
  1. Real Kaggle notebook URL in Cell 0 (markdown header)
  2. Real GitHub repo clone URL in Setup cell
Run once: python scripts/patch_notebook_url.py
"""
import json
from pathlib import Path

KAGGLE_URL = "https://www.kaggle.com/code/thrinainiaroori/finetune"
GITHUB_URL = "https://github.com/thriniiiiiiiiiiii/VLM-Temporal-Operation-Intelligence-for-Logistics.git"
CLONE_DEST = "/kaggle/working/repo"
NB_PATH    = Path(__file__).parent.parent / "finetune.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Patch Cell 0: Markdown header - insert real Kaggle URL
cell0 = nb["cells"][0]
new_source = []
for line in cell0["source"]:
    if "REPLACE WITH YOUR KAGGLE" in line or "YOUR_USERNAME" in line:
        continue
    new_source.append(line)

insert_idx = 1
new_source.insert(insert_idx,     "\n**Live Kaggle Notebook (GPU T4 x2):**\n")
new_source.insert(insert_idx + 1, f"{KAGGLE_URL}\n")
new_source.insert(insert_idx + 2, "\n")
cell0["source"] = new_source

# Patch ALL code cells: fix YOUR_USERNAME git clone placeholder
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    new_lines = []
    for line in cell["source"]:
        if "YOUR_USERNAME" in line and "git clone" in line:
            line = f"!git clone {GITHUB_URL} {CLONE_DEST}\n"
        new_lines.append(line)
    cell["source"] = new_lines

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[OK] finetune.ipynb patched:")
print(f"     Kaggle URL : {KAGGLE_URL}")
print(f"     GitHub URL : {GITHUB_URL}")
