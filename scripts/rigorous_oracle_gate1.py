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
    pred = probs.argmax(axis=1)
    for p in np.unique(prompts):
        idx = np.where(prompts == p)[0]
        n_p = len(idx)
        if 1 < n_p <= probs.shape[1]:
            cost = -probs[idx]
            r, c = linear_sum_assignment(cost)
            pred[idx[r]] = c
    return pred

def paired_bootstrap(y_true, pred_a, pred_b, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        sample_idx = rng.randint(0, n, size=n)
        s_a = balanced_accuracy_score(y_true[sample_idx], pred_a[sample_idx]) * 100
        s_b = balanced_accuracy_score(y_true[sample_idx], pred_b[sample_idx]) * 100
        diffs.append(s_b - s_a)
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), np.mean(diffs <= 0)

def main():
    t0 = time.perf_counter()
    print("=" * 65)
    print("RE-AUDIT OF GATE 1: PROPER COMPETITION-STRUCTURED ORACLE TEST")
    print("=" * 65)

    df = pd.read_parquet("data/processed/train_folds.parquet")
    
    # Strictly select prompts matching the competition structure:
    # 1. Every model appears AT MOST ONCE per prompt (count == nunique)
    # 2. At least 3 distinct models per prompt (meaning multi-model sibling signal exists)
    g = df.groupby("prompt_id")["model"].agg(["count", "nunique"])
    clean_pids = g[(g["count"] == g["nunique"]) & (g["count"] >= 3)].index
    df_clean = df[df["prompt_id"].isin(clean_pids)].copy()

    # Drop classes with fewer than 5 total samples to ensure valid stratified k-fold
    counts = df_clean["model"].value_counts()
    valid_classes = counts[counts >= 5].index
    df_clean = df_clean[df_clean["model"].isin(valid_classes)].copy()
    
    # Re-verify that each prompt still has strictly unique models
    g2 = df_clean.groupby("prompt_id")["model"].agg(["count", "nunique"])
    clean_pids2 = g2[(g2["count"] == g2["nunique"]) & (g2["count"] >= 2)].index
    df_clean = df_clean[df_clean["prompt_id"].isin(clean_pids2)].copy()

    print(f"Verified Clean Slice: {len(df_clean)} passages across {df_clean['prompt_id'].nunique()} prompts.")
    print(f"Models evaluated: {df_clean['model'].nunique()} classes (all prompts have 0 duplicate models).")
    print(f"Passage count per prompt: min={df_clean.groupby('prompt_id').size().min()}, median={df_clean.groupby('prompt_id').size().median()}, max={df_clean.groupby('prompt_id').size().max()}")

    texts = df_clean["text"].fillna("").tolist()
    y_str = df_clean["model"].to_numpy()
    prompt_ids = df_clean["prompt_id"].to_numpy()

    print("\n[Step 1/2] Extracting TF-IDF Features...", flush=True)
    fe = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=40000, sublinear_tf=True)),
        ("char", TfidfVectorizer(ngram_range=(2, 4), analyzer="char", max_features=40000, sublinear_tf=True)),
    ])
    X = fe.fit_transform(texts)
    print(f"  Matrix shape: {X.shape} extracted in {time.perf_counter() - t0:.2f}s")

    lbls = sorted(df_clean["model"].unique())
    lbl2id = {l: i for i, l in enumerate(lbls)}
    y_int = np.array([lbl2id[y] for y in y_str])

    print("\n[Step 2/2] Running 5-Fold Grouped Cross-Validation...", flush=True)
    t_cv = time.perf_counter()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(y_int), len(lbls)), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y_int, prompt_ids)):
        X_tr, y_tr = X[tr_idx], y_int[tr_idx]
        X_va = X[va_idx]

        svc = LinearSVC(C=1.5, max_iter=2000, random_state=42)
        ridge = RidgeClassifier(alpha=1.0, random_state=42)
        svc.fit(X_tr, y_tr)
        ridge.fit(X_tr, y_tr)

        m = 0.70 * svc.decision_function(X_va) + 0.30 * ridge.decision_function(X_va)
        exp_m = np.exp(m / 0.20 - np.max(m / 0.20, axis=1, keepdims=True))
        p = exp_m / np.sum(exp_m, axis=1, keepdims=True)
        oof_probs[va_idx] = p

    print(f"  5-fold CV completed in {time.perf_counter() - t_cv:.2f}s")

    # Evaluation
    pred_argmax = oof_probs.argmax(axis=1)
    acc_argmax = balanced_accuracy_score(y_int, pred_argmax) * 100

    pred_hungarian = oracle_assign(oof_probs, prompt_ids)
    acc_hungarian = balanced_accuracy_score(y_int, pred_hungarian) * 100

    delta = acc_hungarian - acc_argmax
    boot_mean, ci_lo, ci_hi, p_val = paired_bootstrap(y_int, pred_argmax, pred_hungarian)

    print("\n" + "=" * 65)
    print("GATE 1 PROPER AUDIT RESULTS")
    print("=" * 65)
    print(f"Argmax Balanced Accuracy:         {acc_argmax:.2f}%")
    print(f"Oracle Hungarian Balanced Acc:     {acc_hungarian:.2f}%")
    print(f"Oracle Assignment Lift (Delta):    {delta:+.2f}% (Bootstrap 95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%], p={p_val:.4f})")
    print("=" * 65)

    if ci_lo > 0 and delta >= 1.5:
        print(f"PROPER VERDICT: GATE 1 PASSED WITH STATISTICAL SIGNIFICANCE (Delta = {delta:+.2f}% > 0, CI > 0).")
        print("Sibling assignment DOES provide a real positive lift when tested on true 1-answer-per-model prompts!")
    else:
        print(f"PROPER VERDICT: GATE 1 CONFIRMED FAILED (Delta = {delta:+.2f}%, CI contains zero).")
        print("Sibling assignment fails even under clean 1-answer-per-model prompt structure.")

if __name__ == "__main__":
    main()

