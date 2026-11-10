import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import re
import time
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

# Load sample dataset
train_df = pd.read_csv("data/sample/train.csv")
print(f"Loaded train_df: {train_df.shape}, Classes: {train_df['model'].nunique()}")

from notebooks.kaggle_tri_ensemble import (
    LABELS, make_prompt_groups, build_cpu_feature_extractor, DualLinearClassifier,
    margins_to_probs, sinkhorn_knopp_balance
)

# Benchmark Baseline DualLinear on 5 Folds
groups = make_prompt_groups(train_df)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_texts = train_df["text"].tolist()
y_str = train_df["model"].to_numpy()

feature_extractor = build_cpu_feature_extractor(style_weight=0.2)
X_all = feature_extractor.fit_transform(all_texts)

scores_baseline = []
scores_trilinear = []
scores_calibrated = []

for fold, (tr_idx, va_idx) in enumerate(sgkf.split(all_texts, y_str, groups)):
    X_tr, y_tr = X_all[tr_idx], y_str[tr_idx]
    X_va, y_va = X_all[va_idx], y_str[va_idx]

    # Model 1: Baseline DualLinear
    clf_dual = DualLinearClassifier(c_svc=1.2, alpha_ridge=1.0, svc_weight=0.7, seed=42, canonical_labels=LABELS)
    clf_dual.fit(X_tr, y_tr)
    m_dual = clf_dual.decision_function(X_va)
    pred_dual = [LABELS[k] for k in np.argmax(m_dual, axis=1)]
    score_dual = balanced_accuracy_score(y_va, pred_dual)
    scores_baseline.append(score_dual)

    # Model 2: TriLinear (LinearSVC + Ridge + LogisticRegression)
    clf_log = LogisticRegression(C=1.5, max_iter=300, random_state=42, solver="lbfgs")
    clf_log.fit(X_tr, y_tr)
    # Align log_reg decision function
    c_to_idx = {c: i for i, c in enumerate(clf_log.classes_)}
    m_log = np.full((X_va.shape[0], len(LABELS)), -999.0, dtype=np.float64)
    raw_log = clf_log.decision_function(X_va)
    for col, lab in enumerate(LABELS):
        if lab in c_to_idx:
            m_log[:, col] = raw_log[:, c_to_idx[lab]]

    # Weighted blend: 0.50 SVC + 0.30 Ridge + 0.20 LogReg
    m_tri = 0.85 * m_dual + 0.15 * m_log
    pred_tri = [LABELS[k] for k in np.argmax(m_tri, axis=1)]
    score_tri = balanced_accuracy_score(y_va, pred_tri)
    scores_trilinear.append(score_tri)

    # Model 3: TriLinear + Sinkhorn Balancing on validation
    probs_tri = margins_to_probs(m_tri, temp=0.20)
    probs_bal = sinkhorn_knopp_balance(probs_tri)
    pred_bal = [LABELS[k] for k in np.argmax(probs_bal, axis=1)]
    score_bal = balanced_accuracy_score(y_va, pred_bal)
    scores_calibrated.append(score_bal)

print(f"Baseline DualLinear 5-Fold CV:   {np.mean(scores_baseline)*100:.2f}%")
print(f"TriLinear (+LogReg) 5-Fold CV:   {np.mean(scores_trilinear)*100:.2f}%")
print(f"Calibrated Sinkhorn 5-Fold CV:   {np.mean(scores_calibrated)*100:.2f}%")
