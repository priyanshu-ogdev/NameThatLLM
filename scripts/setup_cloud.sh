#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Kaggle-LLM-Challenge
python -m venv /workspace/venvs/name-that-llm
/workspace/venvs/name-that-llm/bin/python -m pip install -r requirements.lock
/workspace/venvs/name-that-llm/bin/python -m pip install --no-deps -e .
/workspace/venvs/name-that-llm/bin/python -m pip check
/workspace/venvs/name-that-llm/bin/python -m pytest -q
