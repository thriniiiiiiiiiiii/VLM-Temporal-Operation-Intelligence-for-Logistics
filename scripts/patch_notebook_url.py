import json
from pathlib import Path

notebook_path = Path("finetune.ipynb")
if notebook_path.exists():
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # Target the first markdown cell
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            for i, line in enumerate(source):
                if "[REPLACE WITH YOUR KAGGLE NOTEBOOK PUBLIC URL]" in line:
                    source[i] = line.replace("[REPLACE WITH YOUR KAGGLE NOTEBOOK PUBLIC URL]", "[Kaggle Notebook Public URL]")
                if "YOUR_USERNAME/vlm-openpack-finetune" in line:
                    source[i] = line.replace("YOUR_USERNAME/vlm-openpack-finetune", "thrinainiaroori/finetune")
            break
            
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("✅ Notebook patched successfully.")
else:
    print("❌ finetune.ipynb not found.")
