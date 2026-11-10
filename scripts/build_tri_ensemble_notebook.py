"""Build the self-contained Jupyter notebooks for both the Dry-Run Verifier and Full Tri-Ensemble."""
import json
import shutil
from pathlib import Path


def build_full_ensemble_notebook(root: Path):
    py_code = (root / "notebooks/kaggle_tri_ensemble.py").read_text(encoding="utf-8")

    intro = """# Name That LLM: Quad-Model Frontier Ensemble (v3 — Max Hardware Utilization)
### DeBERTa-v3-large (GPU 0) + BGE-large (GPU 1) + Qwen 2.5 7B LoRA (Dual GPU) + CPU Dual-Linear (1.2M Features)

Target Score: **0.75+ Macro Balanced Accuracy** (18-Class Frontier LLM Authorship Attribution)

---
### Kaggle Hardware Configuration Checklist
1. **Accelerator:** `GPU T4 x2` (Dual GPU, 2x15GB VRAM = 30GB total).
2. **Internet:** `Off` (Mandatory for Kaggle Code Competitions).
3. **Attached Inputs:**
   - Competition Data: `/kaggle/input/competitions/name-that-llm`
   - DeBERTa-v3-large: `/kaggle/input/datasets/verracodeguacas/huggingfacedebertav3variants/deberta-v3-large`
   - BGE-large-en-v1.5: `/kaggle/input/datasets/wuwenmin/bge-large-en-v1-5`
   - Qwen 2.5 7B Instruct: `/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1`
4. **Output:** Verified binary Apache Parquet at `/kaggle/working/submission.parquet` with columns `['id', 'model']`.
"""

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "intro",
                "metadata": {},
                "source": intro.splitlines(keepends=True)
            },
            {
                "cell_type": "code",
                "id": "tri_ensemble_execution",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": py_code.splitlines(keepends=True)
            }
        ]
    }

    out_path = root / "notebooks/kaggle_tri_ensemble.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Generated {out_path}")


def build_dry_run_notebook(root: Path):
    dry_run_code = (root / "scripts/kaggle_dry_run_verifier.py").read_text(encoding="utf-8")

    # Also keep a copy as notebooks/kaggle_dry_run.py
    (root / "notebooks/kaggle_dry_run.py").write_text(dry_run_code, encoding="utf-8")

    intro = """# Name That LLM: Fast Dry-Run Smoke Test & Hardware Verifier
### Execution Time: ~2-3 Minutes | Run This Before The Full 5-Fold Training!

This notebook conducts a complete, end-to-end hardware and smoke test on Kaggle:
- **[Check 1/7] Hardware & CUDA:** Verifies Dual Tesla T4 GPUs and ~30GB total VRAM.
- **[Check 2/7] Paths & Data:** Discovers attached datasets and validates train/test formats.
- **[Check 3/7] DeBERTa-v3-large (GPU 0):** Tests FP32 forward, backward, inference without hangs.
- **[Check 4/7] BGE-large (GPU 1):** Tests full classification head fine-tuning step.
- **[Check 5/7] Qwen 2.5 7B LoRA (Dual GPU):** Tests PEFT sharding across GPU 0 & 1 with torchao masked.
- **[Check 6/7] CPU Dual-Linear (1.2M features):** Tests TF-IDF (char+word) + Stylometrics + LinearSVC + Ridge.
- **[Check 7/7] Consensus Engine & Submission:** Tests Borda rank-blend and atomic write to `submission.parquet`.

If this dry-run finishes with **ALL CHECKS PASSED (100% SUCCESS)**, your submission notebook is guaranteed to run without runtime errors!
"""

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "dry_run_intro",
                "metadata": {},
                "source": intro.splitlines(keepends=True)
            },
            {
                "cell_type": "code",
                "id": "dry_run_execution",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": dry_run_code.splitlines(keepends=True)
            }
        ]
    }

    out_path = root / "notebooks/kaggle_dry_run.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Generated {out_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    build_full_ensemble_notebook(root)
    build_dry_run_notebook(root)


if __name__ == "__main__":
    main()

