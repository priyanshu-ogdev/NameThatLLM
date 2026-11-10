# Standalone Kaggle CPU submission: attach competition data, Internet Off, accelerator None.
import hashlib
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


# The environment override is for local verification; Kaggle needs no edits.
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
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    return s.strip()

train = pd.read_csv(data_dir/'train.csv', dtype={'id': str, 'prompt_id': str}, keep_default_na=False)
test = pd.read_csv(data_dir/'test.csv', dtype={'id': str}, keep_default_na=False)
train['text'] = train['text'].map(clean_text)
test['text'] = test['text'].map(clean_text)
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']

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


class RichStyleFeatures(TransformerMixin, BaseEstimator):
    """Counts, rates, and length summaries; no fitted vocabulary or prompt labels."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        punctuation = ['—', '–', '-', ',', '.', ';', ':', '!', '?', '…', '...',
                       '"', "'", '’', '“', '”', '(', ')', '[', ']', '*', '#', '\n', '\t', '/', '%']
        for text in X:
            words = re.findall(r"\b\w+(?:['’]\w+)?\b", text)
            lower = [word.lower() for word in words]
            lengths = np.array([len(word) for word in words] or [0])
            sentences = [s for s in re.split(r'[.!?]+\s*', text) if s.strip()]
            sentence_lengths = np.array([len(s.split()) for s in sentences] or [0])
            nw, nc = max(len(words), 1), max(len(text), 1)
            counts = np.array([text.count(p) for p in punctuation], dtype=float)
            function_words = 'the a an and or but if then so because although however therefore thus moreover furthermore indeed perhaps might could would should may must can will we you i it they this that these those not no very more most such'.split()
            rows.append([
                np.log1p(nc), np.log1p(nw), np.log1p(len(sentences)),
                float(lengths.mean()), float(lengths.std()), float(np.median(lengths)),
                float(sentence_lengths.mean()), float(sentence_lengths.std()),
                len(set(lower)) / nw, sum(len(w) <= 3 for w in words) / nw,
                sum(len(w) >= 8 for w in words) / nw,
                sum(c.isupper() for c in text) / nc,
                sum(c.isdigit() for c in text) / nc,
                sum(ord(c) > 127 for c in text) / nc,
                len(re.findall(r"\b\w+['’]\w+\b", text)) / nw,
                *np.log1p(counts), *(counts / nw),
                *[lower.count(word) / nw for word in function_words],
                # Positional Slicing (Openings & Closings)
                float(bool(re.search(r"^\s*#{1,6}\s", text))),
                float(bool(re.search(r"^\s*[-*•]\s|^\s*\d+\.\s", text))),
                float(bool(re.search(r"^\s*\*\*", text))),
                float(bool(re.search(r"^(certainly|sure|here is|as an|to answer|i would)", " ".join(words[:max(nw // 4, 1)]), re.I))),
                float(bool(re.search(r"(overall|in conclusion|in summary|ultimately|hope this helps|to summarize)", " ".join(words[-max(nw // 4, 1):]), re.I))),
                float(text.endswith("?")),
                float(text.endswith("!")),
            ])
        return np.asarray(rows, dtype=np.float64)

def advanced_features(name):
    """Initial feature families declared before evaluating their outer folds."""
    style = Pipeline([("extract", RichStyleFeatures()), ("scale", StandardScaler())])
    if name in ("style_rbf", "style_hgb"):
        return style
    char = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                          sublinear_tf=True, min_df=2, max_features=250_000)
    word = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r"\S+",
                          sublinear_tf=True, min_df=2, max_features=150_000)
    if name == "punct_svm":
        char.set_params(analyzer="char_wb", max_features=200_000)
        word.set_params(token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    branches = [("char", char), ("word", word), ("style", style)]
    if name == "multiview_svm":
        char.set_params(max_features=180_000)
        branches.extend([
            ("boundary", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                  sublinear_tf=True, max_features=100_000)),
            ("punct", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2),
                  lowercase=False, sublinear_tf=True, min_df=2, max_features=100_000)),
        ])
    return FeatureUnion(branches, transformer_weights={"style": 0.2}, n_jobs=1)

def build_submission_model():
    features = advanced_features('multiview_svm')
    features.set_params(transformer_weights={'style': 0.2})
    return Pipeline([
        ('features', features),
        ('classifier', LinearSVC(C=1.5, class_weight='balanced', dual=True,
                                 max_iter=15000, random_state=42)),
    ])

model = build_submission_model()


started = time.perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', ConvergenceWarning)
    with threadpool_limits(limits=2):
        model.fit(train.text, train.model)
for warning in caught:
    print('Training warning:', warning.message)
training_seconds = time.perf_counter() - started
print(f'Full-data model trained in {training_seconds:.1f} seconds. No GPU was used.')


def sinkhorn_knopp_balance(scores: np.ndarray, temperature: float = 1.0, max_iters: int = 50, tol: float = 1e-3) -> np.ndarray:
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
(work/'training_info.json').write_text(json.dumps(run_info, indent=2)+'\n')
print(f'SUCCESS: validated submission saved to {output}')
print('Submission rows:', len(submission))
print(submission.head(10).to_string(index=False))
