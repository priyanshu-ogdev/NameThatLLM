"""Strict schema checks and reusable, duplicate-aware prompt folds."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Confirmed against the user's competition train.csv (SHA-256 recorded in runs).
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_data(path, train=True):
    # Read IDs as strings to preserve leading zeroes, and literal 'NA' as text.
    df = pd.read_csv(path, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    required = ["id", "text"] + (["prompt_id", "model"] if train else [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    for col in required:
        if df[col].isna().any() or not df[col].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{col} must contain nonempty strings")
    if df.id.duplicated().any():
        raise ValueError("Duplicate IDs are not allowed")
    if train and set(df.model) != set(LABELS):
        raise ValueError(f"Expected all 18 labels. Missing={set(LABELS)-set(df.model)}, unknown={set(df.model)-set(LABELS)}. Verify the official label list.")
    return df


def text_keys(df):
    # Conservative duplicate detection; original text stays untouched for modeling.
    return df.text.map(lambda s: hashlib.sha256(" ".join(s.split()).encode()).hexdigest())


def audit(df):
    keys = text_keys(df)
    words = df.text.str.split().str.len()
    grouped = df.assign(text_key=keys).groupby("text_key")
    return {
        "rows": len(df), "prompts": int(df.prompt_id.nunique()),
        "class_counts": {k: int(v) for k, v in df.model.value_counts().items()},
        "word_count_quantiles": {str(k): float(v) for k, v in words.quantile([0,.1,.5,.9,1]).items()},
        "duplicate_text_rows": int(keys.duplicated(keep=False).sum()),
        "texts_with_multiple_labels": int((grouped.model.nunique() > 1).sum()),
        "texts_with_multiple_prompts": int((grouped.prompt_id.nunique() > 1).sum()),
        "note": "Duplicate-connected prompts share one fold; no text is silently dropped.",
    }


def groups(df):
    parent = {p: p for p in df.prompt_id.unique()}
    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    seen = {}
    for prompt, key in zip(df.prompt_id, text_keys(df)):
        if key in seen:
            parent[root(prompt)] = root(seen[key])
        else:
            seen[key] = prompt
    return np.array([root(p) for p in df.prompt_id])


def make_folds(df, n_splits=5, seed=42):
    group_ids = groups(df)
    if len(set(group_ids)) < n_splits:
        raise ValueError("Too few independent prompt groups for requested folds")
    folds = np.full(len(df), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train, valid) in enumerate(splitter.split(df.text, df.model, group_ids)):
        if set(df.model.iloc[train]) != set(LABELS) or set(df.model.iloc[valid]) != set(LABELS):
            raise ValueError("Every train/validation fold must contain all 18 labels; reduce folds or inspect data")
        folds[valid] = fold
    return pd.DataFrame({"id": df.id, "fold": folds})


def validate_folds(df, fold_df):
    if list(fold_df.columns) != ["id", "fold"] or fold_df.id.duplicated().any():
        raise ValueError("Fold file needs unique id and fold columns")
    if set(fold_df.id) != set(df.id):
        raise ValueError("Fold IDs do not match training IDs")
    folds = fold_df.set_index("id").loc[df.id, "fold"].to_numpy()
    if not np.issubdtype(folds.dtype, np.integer) or set(folds) != set(range(len(set(folds)))) or len(set(folds)) < 2:
        raise ValueError("Fold values must be consecutive integers starting at zero")
    for values in (df.prompt_id, groups(df)):
        if pd.DataFrame({"group": np.asarray(values), "fold": folds}).groupby("group").fold.nunique().max() != 1:
            raise ValueError("Prompt or duplicate text leakage across folds")
    for f in set(folds):
        if set(df.model[folds == f]) != set(LABELS) or set(df.model[folds != f]) != set(LABELS):
            raise ValueError("Every fold must contain all labels")
    return folds
