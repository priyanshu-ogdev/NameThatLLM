#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Kaggle-LLM-Challenge
python -m venv /workspace/venvs/ntl-embeddings
/workspace/venvs/ntl-embeddings/bin/python -m pip install -r requirements-embeddings.lock
/workspace/venvs/ntl-embeddings/bin/python -m pip install --no-deps -e .
/workspace/venvs/ntl-embeddings/bin/python -m pip check
python scripts/download_encoder.py --out /workspace/models/bge-small-en-v1.5
