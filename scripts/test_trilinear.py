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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

train_df = pd.read_csv("data/sample/train.csv")
groups = make_prompt_groups(train_df)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_texts = train_df["text"].tolist()
y_str = train_df["model"].to_numpy()

feature_extractor = build_cpu_feature_extractor(style_weight=0.2)
X_all = feature_extractor.fit_transform(all_texts)

class TriLinearClassifier:
    def __init__(self, c_svc=1.2, alpha_ridge=1.0, c_log=1.5, canonical_labels=LABELS):
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
        # Weighted Tri-Linear consensus
        return 0.50 * m_s + 0.30 * m_r + 0.20 * m_l

    def _align(self, margins, classes):
        c_to_idx = {c: i for i, c in enumerate(classes)}
        aligned = np.full((margins.shape[0], len(self.canonical_labels)), -999.0, dtype=np.float64)
        for col, lab in enumerate(self.canonical_labels):
            if lab in c_to_idx:
                aligned[:, col] = margins[:, c_to_idx[lab]]
        return aligned

scores = []
for fold, (tr_idx, va_idx) in enumerate(sgkf.split(all_texts, y_str, groups)):
    clf = TriLinearClassifier()
    clf.fit(X_all[tr_idx], y_str[tr_idx])
    m = clf.decision_function(X_all[va_idx])
    p = margins_to_probs(m, temp=0.20)
    p_bal = sinkhorn_knopp_balance(p)
    preds = [LABELS[k] for k in np.argmax(p_bal, axis=1)]
    score = balanced_accuracy_score(y_str[va_idx], preds)
    scores.append(score)

print(f"TriLinear + Sinkhorn 5-Fold Balanced Accuracy: {np.mean(scores)*100:.2f}%")

