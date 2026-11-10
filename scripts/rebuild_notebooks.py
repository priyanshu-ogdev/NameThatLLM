import json
import ast
from pathlib import Path

def rebuild_notebook(py_path, ipynb_path, title):
    py_path = Path(py_path)
    ipynb_path = Path(ipynb_path)
    code_text = py_path.read_text(encoding="utf-8")
    
    # Verify AST syntax first
    ast.parse(code_text)
    
    lines = [line + "\n" for line in code_text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    elif lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
        
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "Production pipeline for Kaggle 'Name That LLM' competition.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            }
        ],
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "dockerImageVersionId": 30805,
                "isGpuEnabled": True,
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"Rebuilt {ipynb_path} from {py_path}: Cell 0 (Markdown header), Cell 1 (Code, {len(lines)} lines)")

if __name__ == "__main__":
    rebuild_notebook(
        "notebooks/kaggle_dry_run.py",
        "notebooks/kaggle_dry_run.ipynb",
        "Name That LLM: Kaggle Production Smoke Test & Model Throughput Benchmark (V6)"
    )
    rebuild_notebook(
        "notebooks/kaggle_tri_ensemble.py",
        "notebooks/kaggle_tri_ensemble.ipynb",
        "Name That LLM: Quad-Model Frontier Ensemble (V6 Production Pipeline)"
    )
    print("Both notebooks rebuilt and verified successfully!")
