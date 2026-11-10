"""Generate a self-contained offline combined submission from tested sources."""
import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    gen = (root / 'src/ntl/generative.py').read_text()
    models = (root / 'src/ntl/models.py').read_text()
    definitions = {node.name: ast.get_source_segment(models, node) for node in ast.parse(models).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    combined = (root / 'src/ntl/combined.py').read_text()
    combined = '\n'.join(line for line in combined.splitlines() if not line.startswith('from ntl.'))
    extra = '''import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
'''
    code = gen + '\n' + extra + '\n' + definitions['RichStyleFeatures'] + '\n\n' + definitions['advanced_features'] + '\n\n' + combined + '\n\nrun_combined()\n'
    intro = '''# Name That LLM — CPU + frozen Qwen rank blend

Attach competition data AND official qwen-lm/qwen-3/transformers/1.7b-base/1.
Enable a free GPU, Internet Off. Paste/run this notebook alone.
The fixed rank rule measured 40.93% on existing validation predictions; Qwen
correctly identified 417 CPU errors. These are historical validation results,
not a leaderboard guarantee. This run repeats five-fold validation with the two
fixed models, refits on all labeled rows, and writes submission.parquet.
No fine-tuning, API calls, extra data or ensemble weight search is performed.
Save combined-v1/summary.json and validation.json after SUCCESS, then stop the
interactive GPU session. For submission, Save Version → Save & Run All, then
Submit to Competition. A fresh saved run may repeat extraction and GPU usage.
'''
    (root / 'notebooks/kaggle_combined.py').write_text(code)
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
            {'cell_type': 'markdown', 'id': 'instructions', 'metadata': {}, 'source': intro.splitlines(keepends=True)},
            {'cell_type': 'code', 'id': 'combined', 'metadata': {}, 'execution_count': None,
             'outputs': [], 'source': code.splitlines(keepends=True)}]}
    (root / 'notebooks/kaggle_combined.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')


if __name__ == '__main__':
    main()
