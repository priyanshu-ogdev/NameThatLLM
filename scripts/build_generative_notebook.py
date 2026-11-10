"""Embed the tested frozen pipeline in an offline Kaggle notebook and Python twin."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/ntl/generative.py").read_text()
    introduction = """# Frozen Qwen3-1.7B-Base — Name That LLM

Attach competition data and the official **qwen-lm/qwen-3 → Transformers →
1.7b-base → version 1** model. Enable a **free GPU**; set **Internet: Off**.
No API keys, pip installs, or model ZIP uploads are needed.

The transformer stays frozen. Four feature views and two SVM heads are compared
on the original five prompt-grouped folds. All scaling/classifier training is
fold-local. The selected head is retrained on all labeled rows to predict test.csv.
It writes submission.parquet plus qwen-frozen-v1/{summary.json,comparison.csv}.

This is an experiment: improvement over the CPU baseline is not established.
Share the final RESULT TO SHARE block before deciding whether to submit.
Features checkpoint after each batch; Run All can resume in the same session.
A fresh Save & Run All session may recompute them and consumes additional GPU time.
Stop the interactive session when finished to avoid idle quota use.
"""
    script = "# Offline Kaggle GPU experiment: attach competition data + official Qwen3-1.7B-Base v1.\n" + source + "\n\nrun_kaggle()\n"
    (root / "notebooks/kaggle_qwen_frozen.py").write_text(script)
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "cells": [{"cell_type": "markdown", "id": "instructions", "metadata": {}, "source": introduction.splitlines(keepends=True)},
                          {"cell_type": "code", "id": "frozen-experiment", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": script.splitlines(keepends=True)}]}
    (root / "notebooks/kaggle_qwen_frozen.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
