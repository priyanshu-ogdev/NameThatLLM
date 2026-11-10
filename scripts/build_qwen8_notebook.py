"""Compose the standalone Qwen8 pilot/experiment from tested pipeline sources."""
import json
from pathlib import Path

from build_combined_notebook import main as build_combined


def main():
    build_combined()
    root = Path(__file__).resolve().parents[1]
    base = (root / 'notebooks/kaggle_combined.py').read_text().rsplit('\nrun_combined()', 1)[0]
    extension = (root / 'src/ntl/qwen8.py').read_text()
    extension = '\n'.join(line for line in extension.splitlines() if not line.startswith('from ntl.'))
    code = base + '\n\n' + extension + '\n\n_ = run_qwen8()\n'
    intro = '''# Qwen3-8B-Base frozen experiment

New experiment; keep the successful 0.417 notebook unchanged.
Attach competition data and qwen-lm/qwen-3/transformers/8b-base/1.
Select GPU T4 ×2, Internet Off. No pip installs, APIs, quantization or fine-tuning.
The model is distributed across both GPUs. A 16-passage longest-text pilot checks
memory and conservatively projects runtime with 4000 test rows. If over budget,
it stops before full extraction. Default budget is two hours, not a runtime promise.
If it passes, the same original prompt folds, CPU model, Qwen classifier and fixed
rank blend are evaluated, then fitted on all data for dynamic submission output.
Share qwen8-v1/pilot.json, summary.json, validation.json, oof.csv, oof_scores.npz,
per_class.csv and confusion.csv. A pilot failure needs only pilot.json and the log.
Do not submit based on the historical 40.93%: review this experiment's actual results.
Stop the interactive GPU session after saving results. GPU execution is untested
in the CPU-only development workspace; no new competition score is claimed.
'''
    (root / 'notebooks/kaggle_qwen8.py').write_text(code)
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
            {'cell_type': 'markdown', 'id': 'instructions', 'metadata': {}, 'source': intro.splitlines(keepends=True)},
            {'cell_type': 'code', 'id': 'qwen8', 'metadata': {}, 'execution_count': None,
             'outputs': [], 'source': code.splitlines(keepends=True)}]}
    (root / 'notebooks/kaggle_qwen8.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')


if __name__ == '__main__':
    main()
