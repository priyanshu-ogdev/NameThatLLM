import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import time
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from notebooks.kaggle_tri_ensemble import (
    LABELS, make_prompt_groups, build_cpu_feature_extractor,
    DualLinearClassifier, margins_to_probs, sinkhorn_knopp_balance,
    RichStyleFeatures
)

print("=" * 70)
print("TESTING PRECISE CPU UPGRADES AUDIT")
print("=" * 70)

train_df = pd.read_csv("data/sample/train.csv")
all_texts = train_df["text"].tolist()
y_str = train_df["model"].to_numpy()
groups = make_prompt_groups(train_df)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# 1. Feature Extraction
t0 = time.perf_counter()
feature_extractor = build_cpu_feature_extractor(style_weight=0.2)
X_all = feature_extractor.fit_transform(all_texts)
print(f"X_all shape: {X_all.shape} extracted in {time.perf_counter()-t0:.2f}s")

# Extract dense style features alone for tree model
style_ext = RichStyleFeatures()
X_style = style_ext.fit_transform(all_texts)
print(f"X_style dense shape: {X_style.shape}")

# Test LightGBM availability
has_lgb = False
try:
    import lightgbm as lgb
    has_lgb = True
    print("LightGBM is available [OK]")
except ImportError:
    print("LightGBM not installed.")

# Run 5-fold CV comparing:
# A: DualLinear (LinearSVC + Ridge)
# B: TriLinear (LinearSVC + Ridge + LogisticRegression)
# C: TriLinear + Sinkhorn
# D: TriLinear + LightGBM on Dense Style (if available)

scores_a, scores_b, scores_c, scores_d = [], [], [], []

class TriLinearClassifier:
    def __init__(self, c_svc=1.2, alpha_ridge=1.0, c_log=1.0, canonical_labels=LABELS):
        self.svc = LinearSVC(C=c_svc, class_weight="balanced", dual="auto", max_iter=600, tol=1e-3, random_state=42)
        self.ridge = RidgeClassifier(alpha=alpha_ridge, class_weight="balanced", solver="auto", random_state=42)
        self.logreg = LogisticRegression(C=c_log, class_weight="balanced", max_iter=300, solver="lbfgs", random_state=42)
        self.canonical_labels = canonical_labels

    def fit(self, X, y):
        self.svc.fit(X, y)
        self.ridge.fit(X, y)
        self.logreg.fit(X, y)
        return self

    def decision_function(self, X):
        m_s = self._align(self.svc.decision_function(X), self.svc.classes_)
        m_r = self._align(self.ridge.decision_function(X), self.ridge.classes_)
        m_l = self._align(self.logreg.decision_function(X), self.logreg.classes_)
        # Optimal combination: 0.50 SVC + 0.30 Ridge + 0.20 LogReg
        return 0.50 * m_s + 0.30 * m_r + 0.20 * m_l

    def _align(self, margins, classes):
        c_to_idx = {c: i for i, c in enumerate(classes)}
        aligned = np.full((margins.shape[0], len(self.canonical_labels)), -999.0, dtype=np.float64)
        for col, lab in enumerate(self.canonical_labels):
            if lab in c_to_idx:
                aligned[:, col] = margins[:, c_to_idx[lab]]
        return aligned

for fold, (tr_idx, va_idx) in enumerate(sgkf.split(all_texts, y_str, groups)):
    X_tr, y_tr = X_all[tr_idx], y_str[tr_idx]
    X_va, y_va = X_all[va_idx], y_str[va_idx]

    # Model A: DualLinear
    dual = DualLinearClassifier(c_svc=1.2, alpha_ridge=1.0, svc_weight=0.7, seed=42, canonical_labels=LABELS)
    dual.fit(X_tr, y_tr)
    m_a = dual.decision_function(X_va)
    scores_a.append(balanced_accuracy_score(y_va, [LABELS[k] for k in np.argmax(m_a, axis=1)]))

    # Model B: TriLinear
    tri = TriLinearClassifier(canonical_labels=LABELS)
    tri.fit(X_tr, y_tr)
    m_b = tri.decision_function(X_va)
    scores_b.append(balanced_accuracy_score(y_va, [LABELS[k] for k in np.argmax(m_b, axis=1)]))

    # Model C: TriLinear + Sinkhorn
    p_b = margins_to_probs(m_b, temp=0.20)
    p_c = sinkhorn_knopp_balance(p_b)
    scores_c.append(balanced_accuracy_score(y_va, [LABELS[k] for k in np.argmax(p_c, axis=1)]))

    # Model D: LightGBM on Dense Style if available
    if has_lgb:
        lgb_tr = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1)
        label2id = {l: i for i, l in enumerate(LABELS)}
        y_tr_int = np.array([label2id[y] for y in y_tr])
        lgb_tr.fit(X_style[tr_idx], y_tr_int)
        p_lgb = lgb_tr.predict_proba(X_style[va_idx])
        # Blend TriLinear + LightGBM
        p_blend = 0.85 * p_b + 0.15 * p_lgb
        scores_d.append(balanced_accuracy_score(y_va, [LABELS[k] for k in np.argmax(p_blend, axis=1)]))

print("\n--- 5-FOLD CV COMPARISON RESULTS ---")
print(f"  Model A (DualLinear SVC+Ridge):        {np.mean(scores_a)*100:.2f}%")
print(f"  Model B (TriLinear SVC+Ridge+LogReg):  {np.mean(scores_b)*100:.2f}%")
print(f"  Model C (TriLinear + Sinkhorn):        {np.mean(scores_c)*100:.2f}%")
if has_lgb:
    print(f"  Model D (TriLinear + LightGBM Style):  {np.mean(scores_d)*100:.2f}%")
print("Done!")

