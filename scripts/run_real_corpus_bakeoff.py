import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from notebooks.kaggle_tri_ensemble import (
    LABELS, NUM_CLASSES, make_prompt_groups,
    build_cpu_feature_extractor, RichStyleFeatures
)

def compute_prompt_df_mask(Xtr, groups, min_prompts):
    B = (Xtr > 0).astype(np.float32).tocsr()
    codes = pd.factorize(groups)[0]
    G = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))))
    pdf = np.asarray(((G @ B) > 0).sum(axis=0)).ravel()
    return (pdf >= min_prompts).astype(np.float32)

def main():
    print("Loading data/processed/train_folds.parquet...", flush=True)
    df = pd.read_parquet("data/processed/train_folds.parquet")
    print(f"Total rows: {len(df)}, folds: {sorted(df['fold'].unique())}", flush=True)

    all_texts = df["text"].fillna("").tolist()
    y_str = df["model"].to_numpy()
    group_ids = df["prompt_id"].to_numpy()
    fold_ids = df["fold"].to_numpy()

    print("Extracting sparse TF-IDF features (5 branches)...", flush=True)
    t0 = time.perf_counter()
    fe = build_cpu_feature_extractor(style_weight=0.2)
    X_sparse = fe.fit_transform(all_texts)
    print(f"Sparse shape: {X_sparse.shape} in {time.perf_counter() - t0:.2f}s", flush=True)

    print("Extracting dense stylometric features...", flush=True)
    t0 = time.perf_counter()
    V_dense = RichStyleFeatures().fit_transform(all_texts)
    print(f"Dense shape: {V_dense.shape} in {time.perf_counter() - t0:.2f}s", flush=True)

    results = {
        "baseline": [],
        "c_tuned": [],
        "pdf_3": [],
        "pdf_5": [],
        "dense_stream": [],
        "blend_sparse_dense": []
    }

    for fold_i in range(5):
        t_f = time.perf_counter()
        tr_mask = (fold_ids != fold_i)
        va_mask = (fold_ids == fold_i)

        X_tr, y_tr, g_tr = X_sparse[tr_mask], y_str[tr_mask], group_ids[tr_mask]
        X_va, y_va = X_sparse[va_mask], y_str[va_mask]
        V_tr, V_va = V_dense[tr_mask], V_dense[va_mask]

        # 1. Baseline: C=1.2, alpha=1.0
        svc_base = LinearSVC(C=1.2, max_iter=2000, random_state=42)
        ridge_base = RidgeClassifier(alpha=1.0, random_state=42)
        svc_base.fit(X_tr, y_tr)
        ridge_base.fit(X_tr, y_tr)
        m_base = 0.7 * svc_base.decision_function(X_va) + 0.3 * ridge_base.decision_function(X_va)
        pred_base = svc_base.classes_[np.argmax(m_base, axis=1)]
        s_base = balanced_accuracy_score(y_va, pred_base) * 100
        results["baseline"].append(s_base)

        # 2. Tuned regularization: C=0.4, alpha=4.0
        svc_reg = LinearSVC(C=0.4, max_iter=2000, random_state=42)
        ridge_reg = RidgeClassifier(alpha=4.0, random_state=42)
        svc_reg.fit(X_tr, y_tr)
        ridge_reg.fit(X_tr, y_tr)
        m_reg = 0.65 * svc_reg.decision_function(X_va) + 0.35 * ridge_reg.decision_function(X_va)
        pred_reg = svc_reg.classes_[np.argmax(m_reg, axis=1)]
        s_reg = balanced_accuracy_score(y_va, pred_reg) * 100
        results["c_tuned"].append(s_reg)

        # 3. PromptDF mask min_prompts=3
        m3 = compute_prompt_df_mask(X_tr, g_tr, min_prompts=3)
        diag3 = sp.diags(m3)
        X_tr_3 = X_tr @ diag3
        X_va_3 = X_va @ diag3
        svc_p3 = LinearSVC(C=0.4, max_iter=2000, random_state=42)
        svc_p3.fit(X_tr_3, y_tr)
        pred_p3 = svc_p3.classes_[np.argmax(svc_p3.decision_function(X_va_3), axis=1)]
        s_p3 = balanced_accuracy_score(y_va, pred_p3) * 100
        results["pdf_3"].append(s_p3)

        # 4. PromptDF mask min_prompts=5
        m5 = compute_prompt_df_mask(X_tr, g_tr, min_prompts=5)
        diag5 = sp.diags(m5)
        X_tr_5 = X_tr @ diag5
        X_va_5 = X_va @ diag5
        svc_p5 = LinearSVC(C=0.4, max_iter=2000, random_state=42)
        svc_p5.fit(X_tr_5, y_tr)
        pred_p5 = svc_p5.classes_[np.argmax(svc_p5.decision_function(X_va_5), axis=1)]
        s_p5 = balanced_accuracy_score(y_va, pred_p5) * 100
        results["pdf_5"].append(s_p5)

        # 5. Scaled Dense stream
        scaler = StandardScaler()
        V_tr_s = scaler.fit_transform(V_tr)
        V_va_s = scaler.transform(V_va)
        clf_d = LogisticRegression(C=0.2, max_iter=1000, random_state=42)
        clf_d.fit(V_tr_s, y_tr)
        p_d = clf_d.predict_proba(V_va_s)
        pred_d = clf_d.classes_[np.argmax(p_d, axis=1)]
        s_d = balanced_accuracy_score(y_va, pred_d) * 100
        results["dense_stream"].append(s_d)

        # 6. Blend sparse (reg) + dense
        exp_m = np.exp(m_reg / 0.20 - np.max(m_reg / 0.20, axis=1, keepdims=True))
        p_sp = exp_m / np.sum(exp_m, axis=1, keepdims=True)
        # align dense classes if needed
        d_map = [list(clf_d.classes_).index(c) for c in svc_reg.classes_]
        p_d_aligned = p_d[:, d_map]
        p_blend = 0.85 * p_sp + 0.15 * p_d_aligned
        pred_bl = svc_reg.classes_[np.argmax(p_blend, axis=1)]
        s_bl = balanced_accuracy_score(y_va, pred_bl) * 100
        results["blend_sparse_dense"].append(s_bl)

        print(f"Fold {fold_i}: Base={s_base:.2f}% | Reg={s_reg:.2f}% | PDF3={s_p3:.2f}% | PDF5={s_p5:.2f}% | Dense={s_d:.2f}% | Blend={s_bl:.2f}% ({time.perf_counter() - t_f:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("5-FOLD CV SUMMARY ON REAL 20,583 CORPUS")
    print("=" * 60)
    for k, v in results.items():
        arr = np.array(v)
        diff = arr - np.array(results["baseline"])
        print(f"{k:20s}: Mean={arr.mean():.2f}% +/- {arr.std():.2f}% | Delta vs Base={diff.mean():+.2f}%")

if __name__ == "__main__":
    main()
