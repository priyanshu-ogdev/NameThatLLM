"""Verify Jupyter Notebook file validity and structure."""
import json
import ast
from pathlib import Path

def verify_nb(path: Path):
    print(f"Checking notebook: {path.name}")
    content = path.read_text(encoding="utf-8")
    data = json.loads(content)
    assert "cells" in data, "No cells found"
    assert "metadata" in data, "No metadata found"
    print(f"Total cells: {len(data['cells'])}")
    code_cells = [c for c in data['cells'] if c.get("cell_type") == "code"]
    print(f"Code cells: {len(code_cells)}")
    for i, cell in enumerate(code_cells):
        code = "".join(cell.get("source", []))
        # Verify valid Python syntax
        ast.parse(code)
        print(f"  Code cell {i} syntax is 100% valid Python (lines: {len(code.splitlines())})")
    print(f"Notebook {path.name} is 100% valid and ready for Kaggle upload!\n")

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    verify_nb(root / "notebooks/kaggle_dry_run.ipynb")
    verify_nb(root / "notebooks/kaggle_tri_ensemble.ipynb")
    if (root / "notebooks/kaggle_train_and_submit.ipynb").exists():
        verify_nb(root / "notebooks/kaggle_train_and_submit.ipynb")

