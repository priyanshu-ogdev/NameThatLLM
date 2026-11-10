import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from notebooks.kaggle_tri_ensemble import LABELS, NUM_CLASSES

def oracle_assign(probs, prompts):
    """
    Hungarian assignment per prompt group.
    For each prompt with n_passages in [2, 18], solves:
        min sum -log_p(passage_i, class_j) subject to class_j being unique per prompt.
    """
    pred = probs.argmax(axis=1)
    for p in np.unique(prompts):
        idx = np.where(prompts == p)[0]
        n_p = len(idx)
        if 1 < n_p <= probs.shape[1]:
            cost = -probs[idx]
            r, c = linear_sum_assignment(cost)
            pred[idx[r]] = c
    return pred

def main():
    t0 = time.perf_counter()
    print("=" * 65)
    print("GATE 1: ORACLE HUNGARIAN ASSIGNMENT CEILING TEST")
    print("=" * 65)
    print("Testing true prompt-level bipartite matching on real LLM texts...")

    df = pd.read_parquet("data/processed/train_folds.parquet")
    
    # Filter to prompts that have at least 2 models (so sibling structure exists)
    prompt_counts = df.groupby("prompt_id")["model"].nunique()
    multi_prompts = prompt_counts[prompt_counts >= 3].index
    df_multi = df[df["prompt_id"].isin(multi_prompts)].copy()
    
    # Take a balanced slice of up to 2,500 rows across multi-model prompts (fast, ~5s runtime)
    df_slice = df_multi.sample(n=min(2500, len(df_multi)), random_state=42).copy()
    # Ensure all sampled prompts have all their rows in the slice to preserve exact sibling sets
    complete_prompts = df_slice["prompt_id"].unique()
    df_slice = df[df["prompt_id"].isin(complete_prompts)].copy()
    if len(df_slice) > 3000:
        df_slice = df_slice.iloc[:3000].copy()

    print(f"Sampled Dataset: {len(df_slice)} passages across {df_slice['prompt_id'].nunique()} multi-passage prompts.")
    print(f"Passages per prompt distribution: min={df_slice.groupby('prompt_id').size().min()}, median={df_slice.groupby('prompt_id').size().median()}, max={df_slice.groupby('prompt_id').size().max()}")

    texts = df_slice["text"].fillna("").tolist()
    y_str = df_slice["model"].to_numpy()
    prompt_ids = df_slice["prompt_id"].to_numpy()

    # Fast 2-branch TF-IDF (word 1-2 + char 2-4)
    fe = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("char", TfidfVectorizer(ngram_range=(2, 4), analyzer="char", max_features=50000, sublinear_tf=True)),
    ])
    print("\nExtracting features (fast 2-branch union)...", flush=True)
    X = fe.fit_transform(texts)
    print(f"Matrix shape: {X.shape} extracted in {time.perf_counter() - t0:.2f}s")

    lbl2id = {l: i for i, l in enumerate(LABELS)}
    y_int = np.array([lbl2id.get(y, -1) for y in y_str])
    valid_mask = (y_int >= 0)
    X = X[valid_mask]
    y_int = y_int[valid_mask]
    prompt_ids = prompt_ids[valid_mask]

    # 5-Fold Grouped Cross-Validation
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(y_int), NUM_CLASSES), dtype=np.float64)

    print("\nFitting 5-fold CV linear models...", flush=True)
    t_cv = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y_int, prompt_ids)):
        X_tr, y_tr = X[tr_idx], y_int[tr_idx]
        X_va = X[va_idx]

        svc = LinearSVC(C=1.5, max_iter=2000, random_state=42)
        ridge = RidgeClassifier(alpha=1.0, random_state=42)
        svc.fit(X_tr, y_tr)
        ridge.fit(X_tr, y_tr)

        # Margins to probabilities via softmax (temp=0.20)
        m = 0.70 * svc.decision_function(X_va) + 0.30 * ridge.decision_function(X_va)
        exp_m = np.exp(m / 0.20 - np.max(m / 0.20, axis=1, keepdims=True))
        p = exp_m / np.sum(exp_m, axis=1, keepdims=True)
        oof_probs[va_idx] = p

    print(f"5-fold CV completed in {time.perf_counter() - t_cv:.2f}s")

    # Evaluation: Argmax vs Oracle Hungarian Assignment
    pred_argmax = oof_probs.argmax(axis=1)
    acc_argmax = balanced_accuracy_score(y_int, pred_argmax) * 100

    pred_hungarian = oracle_assign(oof_probs, prompt_ids)
    acc_hungarian = balanced_accuracy_score(y_int, pred_hungarian) * 100

    delta = acc_hungarian - acc_argmax

    print("\n" + "=" * 65)
    print("GATE 1 RESULTS SUMMARY")
    print("=" * 65)
    print(f"Argmax Balanced Accuracy:         {acc_argmax:.2f}%")
    print(f"Oracle Hungarian Balanced Acc:     {acc_hungarian:.2f}%")
    print(f"Oracle Assignment Lift (Delta):    {delta:+.2f}%")
    print("=" * 65)

    if delta >= 4.0:
        print(f"VERDICT: GATE 1 PASSED (Delta = {delta:+.2f}% >= +4.0%).")
        print("Sibling assignment provides significant signal on this prompt structure.")
    else:
        print(f"VERDICT: GATE 1 FAILED (Delta = {delta:+.2f}% < +4.0%).")
        print("Prompt-level assignment yields negligible lift; abandon sibling direction.")

if __name__ == "__main__":
    main()

