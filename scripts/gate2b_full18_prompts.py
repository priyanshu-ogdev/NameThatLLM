"""
Critical missing gate: Test BGE clustering purity ONLY on full 18-member prompts
from train. This mirrors the actual competition test set structure.
If those prompts cluster cleanly (purity > 90%), sibling decoding IS viable on test.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedGroupKFold

def main():
    print("=" * 70)
    print("GATE 2b: BGE CLUSTERING ON FULL 18-MODEL PROMPTS ONLY")
    print("(Mirrors actual competition test set structure)")
    print("=" * 70)

    df = pd.read_parquet("data/processed/train_folds.parquet")
    g = df.groupby("prompt_id")["model"].agg(["count", "nunique"])
    
    # Full 18-model prompts where every model is unique (clean 18-way groups)
    full18 = g[(g["count"] == 18) & (g["count"] == g["nunique"])].index
    print(f"\nFull 18-model clean prompts in train: {len(full18)}")
    
    if len(full18) == 0:
        print("NONE found. Falling back to prompts with 15+ unique models.")
        full18 = g[(g["count"] >= 15) & (g["count"] == g["nunique"])].index
        print(f"Prompts with 15+ unique models: {len(full18)}")
    
    df_18 = df[df["prompt_id"].isin(full18)].copy()
    print(f"Total passages: {len(df_18)} across {df_18['prompt_id'].nunique()} prompts")
    print(f"Classes present: {df_18['model'].nunique()}")

    texts = df_18["text"].fillna("").tolist()
    prompt_ids = df_18["prompt_id"].to_numpy()
    pid_codes = pd.factorize(prompt_ids)[0]
    n_prompts = df_18["prompt_id"].nunique()
    
    lbls = sorted(df_18["model"].unique())
    lbl2id = {l: i for i, l in enumerate(lbls)}
    y_int = np.array([lbl2id[y] for y in df_18["model"].to_numpy()])

    # Embed with BGE-small
    print("\nEmbedding with BGE-small...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.eval()
    
    batch_size = 64
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = texts[i:i+batch_size]
            inp = tokenizer(b, padding=True, truncation=True, max_length=256, return_tensors="pt")
            out = model(**inp)
            normed = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=-1)
            embs.append(normed.cpu().numpy())
    E = np.vstack(embs)
    print(f"Embeddings shape: {E.shape}")
    
    # Pairwise cosine similarity stats for 18-member prompts
    print("\n--- Pairwise Cosine Similarity Stats ---")
    same_sims = []
    diff_sims = []
    
    pids = np.unique(prompt_ids)
    for i, p1 in enumerate(pids[:50]):  # check first 50 prompts
        idx1 = np.where(prompt_ids == p1)[0]
        for j, p2 in enumerate(pids[:50]):
            idx2 = np.where(prompt_ids == p2)[0]
            sims = E[idx1] @ E[idx2].T
            if i == j:
                # same prompt, off-diagonal
                n = len(idx1)
                mask = ~np.eye(n, dtype=bool)
                same_sims.extend(sims[mask].tolist())
            elif i < j:
                diff_sims.extend(sims.flatten().tolist())
    
    same_sims = np.array(same_sims)
    diff_sims = np.array(diff_sims)
    print(f"Same-prompt cosine: mean={same_sims.mean():.4f}, std={same_sims.std():.4f}, p5={np.percentile(same_sims, 5):.4f}")
    print(f"Diff-prompt cosine: mean={diff_sims.mean():.4f}, std={diff_sims.std():.4f}, p95={np.percentile(diff_sims, 95):.4f}")
    overlap_lo = np.percentile(same_sims, 5)
    overlap_hi = np.percentile(diff_sims, 95)
    print(f"Overlap zone: [{min(overlap_lo, overlap_hi):.4f}, {max(overlap_lo, overlap_hi):.4f}]")
    
    # Agglomerative clustering sweep
    print("\n--- Clustering quality on full 18-member prompts ---")
    dist_matrix = 1.0 - (E @ E.T)
    np.fill_diagonal(dist_matrix, 0.0)
    
    print(f"{'Threshold':12s} | {'n_clusters':10s} | {'ARI':8s} | {'Mean Purity':12s} | {'>=90% purity':12s}")
    print("-" * 65)
    
    best_ari = 0
    best_thresh = None
    
    for dist_thresh in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        cl = AgglomerativeClustering(
            n_clusters=None, distance_threshold=dist_thresh,
            metric="precomputed", linkage="average"
        ).fit_predict(dist_matrix)
        
        ari = adjusted_rand_score(pid_codes, cl)
        
        # Per-prompt top-k purity
        purities = []
        for p in np.unique(prompt_ids):
            idx = np.where(prompt_ids == p)[0]
            k = len(idx)
            # for each passage, what fraction of its k-1 nearest neighbors are in the same prompt?
            for ii in idx:
                sims_ii = E[ii] @ E.T
                sims_ii[ii] = -1  # exclude self
                top_k = np.argsort(sims_ii)[::-1][:k-1]
                same = sum(prompt_ids[t] == p for t in top_k)
                purities.append(same / (k - 1))
        
        purities = np.array(purities)
        mean_purity = purities.mean()
        frac_90 = (purities >= 0.90).mean()
        
        n_cl = len(np.unique(cl))
        cos_thresh = 1.0 - dist_thresh
        print(f"cos>={cos_thresh:.2f} (d<={dist_thresh:.2f}) | {n_cl:10d} | {ari:8.4f} | {mean_purity:12.4f} | {frac_90:11.1%}")
        
        if ari > best_ari:
            best_ari = ari
            best_thresh = dist_thresh
    
    print(f"\nBest ARI: {best_ari:.4f} at distance threshold {best_thresh:.2f}")
    
    # Now do a full end-to-end test on 18-member prompts
    print("\n--- End-to-End Hungarian Assignment on 18-member prompts ---")
    if len(df_18) >= 100:
        fe = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=40000, sublinear_tf=True)),
            ("char", TfidfVectorizer(ngram_range=(2, 4), analyzer="char", max_features=40000, sublinear_tf=True)),
        ])
        X = fe.fit_transform(texts)
        splitter = StratifiedGroupKFold(n_splits=min(5, n_prompts), shuffle=True, random_state=42)
        oof_probs = np.zeros((len(y_int), len(lbls)), dtype=np.float64)
        
        fold_count = 0
        for tr_idx, va_idx in splitter.split(X, y_int, prompt_ids):
            svc = LinearSVC(C=1.5, max_iter=2000, random_state=42)
            ridge = RidgeClassifier(alpha=1.0, random_state=42)
            svc.fit(X[tr_idx], y_int[tr_idx])
            ridge.fit(X[tr_idx], y_int[tr_idx])
            m = 0.70 * svc.decision_function(X[va_idx]) + 0.30 * ridge.decision_function(X[va_idx])
            exp_m = np.exp(m / 0.20 - np.max(m / 0.20, axis=1, keepdims=True))
            oof_probs[va_idx] = exp_m / np.sum(exp_m, axis=1, keepdims=True)
            fold_count += 1
        
        print(f"CV folds completed: {fold_count}")
        
        acc_argmax = balanced_accuracy_score(y_int, oof_probs.argmax(1)) * 100
        
        # Oracle assignment
        pred_oracle = oof_probs.argmax(1).copy()
        for p in np.unique(prompt_ids):
            idx = np.where(prompt_ids == p)[0]
            if 1 < len(idx) <= len(lbls):
                r, c = linear_sum_assignment(-oof_probs[idx])
                pred_oracle[idx[r]] = c
        acc_oracle = balanced_accuracy_score(y_int, pred_oracle) * 100
        
        print(f"\nBase argmax accuracy (18-member prompts):   {acc_argmax:.2f}%")
        print(f"Oracle Hungarian (18-member prompts):       {acc_oracle:.2f}%  ({acc_oracle - acc_argmax:+.2f}%)")
        
        # Now test unsupervised clustering on these clean prompts
        cl_best = AgglomerativeClustering(
            n_clusters=None, distance_threshold=best_thresh,
            metric="precomputed", linkage="average"
        ).fit_predict(dist_matrix)
        
        pred_unsuper = oof_probs.argmax(1).copy()
        for c_id in np.unique(cl_best):
            idx = np.where(cl_best == c_id)[0]
            if 1 < len(idx) <= len(lbls):
                r, c = linear_sum_assignment(-oof_probs[idx])
                pred_unsuper[idx[r]] = c
        acc_unsuper = balanced_accuracy_score(y_int, pred_unsuper) * 100
        print(f"Unsupervised Hungarian (best BGE thresh):   {acc_unsuper:.2f}%  ({acc_unsuper - acc_argmax:+.2f}%)")
    else:
        print("Not enough data for CV (too few 18-member prompts)")
        print("CONCLUSION: Cannot reliably test on true 18-member subsets.")
        print("The 21 full-18 prompts are too few for stable 5-fold CV.")

if __name__ == "__main__":
    main()

