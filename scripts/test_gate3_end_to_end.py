import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.cluster import AgglomerativeClustering

def run_gate3_end_to_end():
    print("=" * 70)
    print("ADVERSARIAL AUDIT: GATE 3 END-TO-END RECOVERED CLUSTER ASSIGNMENT")
    print("=" * 70)
    print("Measuring EXACT empirical performance of Hungarian assignment on BGE clusters...\n")

    # Load clean slice
    df = pd.read_parquet("data/processed/train_folds.parquet")
    g = df.groupby("prompt_id")["model"].agg(["count", "nunique"])
    clean_pids = g[(g["count"] == g["nunique"]) & (g["count"] >= 3)].index
    df_clean = df[df["prompt_id"].isin(clean_pids)].copy()

    counts = df_clean["model"].value_counts()
    valid_classes = counts[counts >= 5].index
    df_clean = df_clean[df_clean["model"].isin(valid_classes)].copy()
    
    g2 = df_clean.groupby("prompt_id")["model"].agg(["count", "nunique"])
    clean_pids2 = g2[(g2["count"] == g2["nunique"]) & (g2["count"] >= 2)].index
    df_clean = df_clean[df_clean["prompt_id"].isin(clean_pids2)].copy()

    # Subsample 1,500 passages across ~200 prompts for fast evaluation
    rng = np.random.RandomState(42)
    eval_pids = rng.choice(clean_pids2, size=min(180, len(clean_pids2)), replace=False)
    df_eval = df_clean[df_clean["prompt_id"].isin(eval_pids)].copy()

    texts = df_eval["text"].fillna("").tolist()
    y_str = df_eval["model"].to_numpy()
    prompt_ids = df_eval["prompt_id"].to_numpy()
    lbls = sorted(df_eval["model"].unique())
    lbl2id = {l: i for i, l in enumerate(lbls)}
    y_int = np.array([lbl2id[y] for y in y_str])

    print(f"Dataset: {len(df_eval)} passages across {len(eval_pids)} prompts ({len(lbls)} classes)")

    # 1. 5-Fold CV Probabilities
    print("\n[Step 1/3] Computing 5-Fold Grouped CV Probabilities...", flush=True)
    fe = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=40000, sublinear_tf=True)),
        ("char", TfidfVectorizer(ngram_range=(2, 4), analyzer="char", max_features=40000, sublinear_tf=True)),
    ])
    X = fe.fit_transform(texts)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(y_int), len(lbls)), dtype=np.float64)

    for tr_idx, va_idx in splitter.split(X, y_int, prompt_ids):
        svc = LinearSVC(C=1.5, max_iter=2000, random_state=42)
        ridge = RidgeClassifier(alpha=1.0, random_state=42)
        svc.fit(X[tr_idx], y_int[tr_idx])
        ridge.fit(X[tr_idx], y_int[tr_idx])
        m = 0.70 * svc.decision_function(X[va_idx]) + 0.30 * ridge.decision_function(X[va_idx])
        exp_m = np.exp(m / 0.20 - np.max(m / 0.20, axis=1, keepdims=True))
        oof_probs[va_idx] = exp_m / np.sum(exp_m, axis=1, keepdims=True)

    acc_argmax = balanced_accuracy_score(y_int, oof_probs.argmax(axis=1)) * 100
    print(f"  Base Argmax Balanced Accuracy: {acc_argmax:.2f}%")

    # 2. Oracle Hungarian (True Prompts)
    pred_oracle = oof_probs.argmax(axis=1)
    for p in np.unique(prompt_ids):
        idx = np.where(prompt_ids == p)[0]
        if 1 < len(idx) <= len(lbls):
            r, c = linear_sum_assignment(-oof_probs[idx])
            pred_oracle[idx[r]] = c
    acc_oracle = balanced_accuracy_score(y_int, pred_oracle) * 100
    print(f"  Oracle Hungarian (True Prompts): {acc_oracle:.2f}% ({acc_oracle - acc_argmax:+.2f}%)")

    # 3. Embed with BGE for Unsupervised Clustering
    print("\n[Step 2/3] Embedding texts with BGE-small for unsupervised clustering...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.eval()

    batch_size = 64
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i+batch_size]
            inputs = tokenizer(b_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            out = model(**inputs)
            normed = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=-1)
            embs.append(normed.cpu().numpy())
    E = np.vstack(embs)

    dist_matrix = 1.0 - (E @ E.T)
    np.fill_diagonal(dist_matrix, 0.0)

    # 4. Sweep Distance Thresholds and evaluate Unsupervised Hungarian Matching
    print("\n[Step 3/3] Testing End-to-End Hungarian Assignment on Unsupervised BGE Clusters...")
    print("-" * 75)
    print(f"{'Threshold (cos)':18s} | {'Clusters':8s} | {'Median Size':11s} | {'Accuracy':9s} | {'Delta vs Argmax':16s}")
    print("-" * 75)

    thresholds = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    results = []

    for dist_thresh in thresholds:
        cos_thresh = 1.0 - dist_thresh
        cl = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thresh,
            metric="precomputed",
            linkage="average"
        ).fit_predict(dist_matrix)

        # Apply Hungarian assignment on each recovered cluster
        pred_unsupervised = oof_probs.argmax(axis=1).copy()
        
        # Test 1: Blind Hungarian on all clusters with size in [2, K]
        for c_id in np.unique(cl):
            idx = np.where(cl == c_id)[0]
            if 1 < len(idx) <= len(lbls):
                r, c = linear_sum_assignment(-oof_probs[idx])
                pred_unsupervised[idx[r]] = c

        acc_unsupervised = balanced_accuracy_score(y_int, pred_unsupervised) * 100
        delta = acc_unsupervised - acc_argmax
        n_clusters = len(np.unique(cl))
        med_size = pd.Series(cl).value_counts().median()

        print(f"cos >= {cos_thresh:.2f} (d<={dist_thresh:.2f})  | {n_clusters:8d} | {med_size:11.1f} | {acc_unsupervised:8.2f}% | {delta:+15.2f}%")
        results.append((cos_thresh, acc_unsupervised, delta))

    # Test Gated / High-Confidence Fallback: Only assign if cluster size in [5, 12] AND cluster cohesion is high
    print("\n--- TEST: CONFIDENCE-GATED HUNGARIAN ASSIGNMENT ---")
    for min_size in [4, 6]:
        for dist_thresh in [0.12, 0.15]:
            cos_thresh = 1.0 - dist_thresh
            cl = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dist_thresh,
                metric="precomputed",
                linkage="average"
            ).fit_predict(dist_matrix)

            pred_gated = oof_probs.argmax(axis=1).copy()
            assigned_count = 0
            for c_id in np.unique(cl):
                idx = np.where(cl == c_id)[0]
                # Only apply if cluster size is at least min_size and within bounds
                if min_size <= len(idx) <= len(lbls):
                    # Check intra-cluster cohesion (mean pairwise similarity)
                    sub_S = (E[idx] @ E[idx].T)
                    mean_cohesion = (sub_S.sum() - len(idx)) / max(len(idx) * (len(idx) - 1), 1)
                    if mean_cohesion >= cos_thresh:
                        r, c = linear_sum_assignment(-oof_probs[idx])
                        pred_gated[idx[r]] = c
                        assigned_count += len(idx)

            acc_gated = balanced_accuracy_score(y_int, pred_gated) * 100
            delta_gated = acc_gated - acc_argmax
            print(f"Gated (cos>={cos_thresh:.2f}, size>={min_size}): Passages assigned={assigned_count}/{len(texts)} ({assigned_count/len(texts)*100:.1f}%) | Acc={acc_gated:.2f}% | Delta={delta_gated:+.2f}%")

if __name__ == "__main__":
    run_gate3_end_to_end()

