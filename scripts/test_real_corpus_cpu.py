import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from notebooks.kaggle_tri_ensemble import (
    LABELS, make_prompt_groups, build_cpu_feature_extractor, DualLinearClassifier,
    margins_to_probs, sinkhorn_knopp_balance
)
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/processed/train_folds.csv")
print(f"Total rows in train_folds.csv: {len(df)}")

# Filter to the 18 official classes and pick 2699 rows (or first 150 prompt groups)
df = df[df["model"].isin(LABELS)].copy()
unique_prompts = df["prompt_id"].unique()[:150]
df = df[df["prompt_id"].isin(unique_prompts)].copy()
print(f"Filtered to 150 prompts: {len(df)} rows across {df['model'].nunique()} classes")

# StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_texts = df["text"].tolist()
y_str = df["model"].to_numpy()
groups = df["prompt_id"].to_numpy()

feature_extractor = build_cpu_feature_extractor(style_weight=0.2)
X_all = feature_extractor.fit_transform(all_texts)
print(f"X_all shape: {X_all.shape}")

# Test on Fold 0
tr_idx, va_idx = next(sgkf.split(all_texts, y_str, groups))
X_tr, y_tr = X_all[tr_idx], y_str[tr_idx]
X_va, y_va = X_all[va_idx], y_str[va_idx]

# Model 1: Baseline DualLinear (LinearSVC + Ridge)
clf_dual = DualLinearClassifier(c_svc=1.2, alpha_ridge=1.0, svc_weight=0.7, seed=42, canonical_labels=LABELS)
clf_dual.fit(X_tr, y_tr)
m_dual = clf_dual.decision_function(X_va)
pred_dual = [LABELS[k] for k in np.argmax(m_dual, axis=1)]
score_dual = balanced_accuracy_score(y_va, pred_dual)
print(f"Fold 0 DualLinear Balanced Accuracy: {score_dual*100:.2f}%")

# Model 2: DualLinear + Sinkhorn Test Balancing
p_dual = margins_to_probs(m_dual, temp=0.20)
p_bal = sinkhorn_knopp_balance(p_dual)
pred_bal = [LABELS[k] for k in np.argmax(p_bal, axis=1)]
score_bal = balanced_accuracy_score(y_va, pred_bal)
print(f"Fold 0 DualLinear + Sinkhorn Balancing: {score_bal*100:.2f}% (lift: {score_bal*100 - score_dual*100:+.2f}%)")

# Per-class recalls
print("\nPer-class recalls under Sinkhorn balancing:")
for lab in LABELS:
    mask = (y_va == lab)
    if np.sum(mask) > 0:
        acc = np.mean(np.array(pred_bal)[mask] == lab)
        print(f"  {lab:25s}: {acc*100:5.1f}% (N={np.sum(mask)})")

