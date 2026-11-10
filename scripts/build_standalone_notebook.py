"""Generate a Kaggle-native train-and-submit notebook and pasteable Python twin."""
import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    model_source = (root / "src/ntl/models.py").read_text(encoding="utf-8")
    model_tree = ast.parse(model_source)
    definitions = {node.name: ast.get_source_segment(model_source, node)
                   for node in model_tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    data_tree = ast.parse((root / "src/ntl/data.py").read_text(encoding="utf-8"))
    labels = next(ast.literal_eval(node.value) for node in data_tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LABELS" for t in node.targets))
    intro = """# Name That LLM — standalone CPU submission

Attach only the **Name That LLM competition data**. Set **Accelerator: None** and
**Internet: Off**, then Run All and save a **Save & Run All** version to submit.

This notebook trains our selected four-view TF-IDF/style SVM on all available
training passages, then predicts every supplied test row. No model ZIP, custom
dataset, GitHub connection, pip install, GPU, or API key is needed during execution.
The chosen configuration measured 37.56% in local prompt-grouped cross-validation;
this run fits the full dataset and does not calculate a leaderboard score.
"""
    imports = """import hashlib
import importlib.metadata
import json
import os
import re
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

print('Using Kaggle-installed packages:', {
    name: importlib.metadata.version(name)
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow']
})
"""
    data = """# The environment override is for local verification; Kaggle needs no edits.
candidates = [Path(os.environ['NTL_DATA_DIR'])] if 'NTL_DATA_DIR' in os.environ else [
    Path('/kaggle/input/competitions/name-that-llm'), Path('/kaggle/input/name-that-llm')
]
data_dir = next((p for p in candidates if (p/'train.csv').is_file() and (p/'test.csv').is_file()), None)
if data_dir is None:
    for p in Path('/kaggle/input').rglob('train.csv'):
        if (p.parent / 'test.csv').is_file():
            data_dir = p.parent
            break

if data_dir is None:
    raise RuntimeError('Attach the Name That LLM competition using Add Input, then Run All again.')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    s = s.replace("\\u00a0", " ").replace("\\u200b", "").replace("\\ufeff", "")
    s = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]", "", s)
    return s.strip()

train = pd.read_csv(data_dir/'train.csv', dtype={'id': str, 'prompt_id': str}, keep_default_na=False)
test = pd.read_csv(data_dir/'test.csv', dtype={'id': str}, keep_default_na=False)
train['text'] = train['text'].map(clean_text)
test['text'] = test['text'].map(clean_text)
""" + f"LABELS = {labels!r}\n" + """
def validate_input(frame, required, is_test=False):
    if not set(required) <= set(frame.columns) or frame.empty:
        raise ValueError('Missing required columns or empty competition file')
    if frame.id.duplicated().any():
        raise ValueError('Duplicate passage IDs')
    for name in required:
        if is_test and name == 'text':
            continue
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f'Invalid or empty {name}')

validate_input(train, ['id', 'prompt_id', 'text', 'model'], is_test=False)
validate_input(test, ['id', 'text'], is_test=True)
if set(train.model) != set(LABELS):
    raise ValueError('Training labels differ from the expected 18 competition labels')
print(f'Training passages: {len(train)}; prompts: {train.prompt_id.nunique()}; test passages: {len(test)}')
"""
    model = definitions["RichStyleFeatures"] + "\n\n" + definitions["advanced_features"] + """

def build_submission_model():
    features = advanced_features('multiview_svm')
    features.set_params(transformer_weights={'style': 0.2})
    return Pipeline([
        ('features', features),
        ('classifier', LinearSVC(C=1.5, class_weight='balanced', dual=True,
                                 max_iter=15000, random_state=42)),
    ])

model = build_submission_model()
"""
    fit = """started = time.perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', ConvergenceWarning)
    with threadpool_limits(limits=2):
        model.fit(train.text, train.model)
for warning in caught:
    print('Training warning:', warning.message)
training_seconds = time.perf_counter() - started
print(f'Full-data model trained in {training_seconds:.1f} seconds. No GPU was used.')
"""
    predict = """def sinkhorn_knopp_balance(scores: np.ndarray, temperature: float = 1.0, max_iters: int = 50, tol: float = 1e-3) -> np.ndarray:
    N, K = scores.shape
    scaled = scores / max(temperature, 1e-4)
    P = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    target_col = np.full(K, N / K, dtype=np.float64)
    r = np.ones(N, dtype=np.float64)
    c = np.ones(K, dtype=np.float64)
    for _ in range(max_iters):
        c_prev = c.copy()
        c = target_col / (np.dot(r, P) + 1e-12)
        r = 1.0 / (np.dot(P, c) + 1e-12)
        if np.max(np.abs(c - c_prev)) < tol:
            break
    return r[:, None] * P * c[None, :]

def predict_submission(model, frame, work):
    with threadpool_limits(limits=2):
        margins = model.decision_function(frame.text)
    classes = list(model.classes_)
    if len(frame) >= 36:
        balanced_P = sinkhorn_knopp_balance(margins, temperature=1.0)
        predictions = [classes[i] for i in np.argmax(balanced_P, axis=1)]
    else:
        predictions = [classes[i] for i in np.argmax(margins, axis=1)]
    submission = pd.DataFrame({'id': frame.id.to_numpy(), 'model': predictions})
    if len(submission) != len(frame) or submission.id.duplicated().any():
        raise ValueError('Submission must contain every test ID exactly once')
    if submission.isna().any().any() or not set(submission.model) <= set(LABELS):
        raise ValueError('Null prediction or invalid model label')
    if submission.id.tolist() != frame.id.tolist():
        raise ValueError('Submission ID order or values changed')
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    pending = work/f'.submission-{uuid.uuid4().hex}.parquet'
    submission.to_parquet(pending, index=False)
    pd.testing.assert_frame_equal(submission, pd.read_parquet(pending))
    output = work/'submission.parquet'
    pending.replace(output)
    return output, submission

work = Path(os.environ.get('NTL_WORKING', '/kaggle/working'))
output, submission = predict_submission(model, test, work)
run_info = {
    'candidate': 'multi_c150_s020', 'seed': 42, 'training_rows': len(train),
    'training_data_sha256': hashlib.sha256((data_dir/'train.csv').read_bytes()).hexdigest(),
    'test_rows': len(test), 'training_seconds': training_seconds,
    'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow']},
    'note': 'Full training fit; no in-sample accuracy or leaderboard score is claimed.'
}
(work/'training_info.json').write_text(json.dumps(run_info, indent=2)+'\\n')
print(f'SUCCESS: validated submission saved to {output}')
print('Submission rows:', len(submission))
print(submission.head(10).to_string(index=False))
"""
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True), "id": "standalone-intro"}]
    codes = [imports, data, model, fit, predict]
    for i, code in enumerate(codes):
        cells.append({"cell_type": "code", "metadata": {}, "source": code.splitlines(keepends=True),
                      "execution_count": None, "outputs": [], "id": f"standalone-{i}"})
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (root/'notebooks/kaggle_train_and_submit.ipynb').write_text(json.dumps(notebook, indent=1)+'\n', encoding="utf-8")
    companion = '# Standalone Kaggle CPU submission: attach competition data, Internet Off, accelerator None.\n'
    companion += '\n\n'.join(codes)
    (root/'notebooks/kaggle_train_and_submit.py').write_text(companion.rstrip()+'\n', encoding="utf-8")


if __name__ == '__main__':
    main()
