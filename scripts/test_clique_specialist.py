"""
Testing IntraFamilyCliqueSpecialist on synthetic and real sample data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notebooks.kaggle_tri_ensemble import (
    LABELS,
    NUM_CLASSES,
    FAMILY_MAP,
    FAMILY_TO_INDICES,
    LABEL_TO_IDX,
)


class IntraFamilyCliqueSpecialist:
    """
    Trains dedicated specialist linear classifiers within each multi-model family clique
    (Anthropic-5, Google-4, OpenAI-5, Zhipu-2) on the combined competition + external corpus.
    Provides sharp conditional intra-family sibling log-probabilities to break sibling ties.
    """
    def __init__(self, alpha_ridge: float = 1.0, beta_refine: float = 0.25, seed: int = 42):
        self.alpha_ridge = alpha_ridge
        self.beta_refine = beta_refine
        self.seed = seed
        self.specialists_ = {}

    def fit(self, X, y, sample_weight=None):
        y_arr = np.asarray(y)
        for fam, indices in FAMILY_TO_INDICES.items():
            if len(indices) < 2:
                continue
            fam_labels = [LABELS[i] for i in indices]
            fam_mask = np.isin(y_arr, fam_labels)
            if np.sum(fam_mask) < 20:
                continue
            X_fam = X[fam_mask]
            y_fam = y_arr[fam_mask]
            w_fam = sample_weight[fam_mask] if sample_weight is not None else None

            clf = RidgeClassifier(alpha=self.alpha_ridge, random_state=self.seed)
            clf.fit(X_fam, y_fam, sample_weight=w_fam)
            self.specialists_[fam] = (clf, list(clf.classes_))
        return self

    def refine_margins(self, global_margins: np.ndarray, X_eval) -> np.ndarray:
        """
        Refines global margins with conditional intra-family specialist scores:
        z_m <- z_m + beta * sibling_score_m
        """
        if not self.specialists_:
            return global_margins
        refined = np.copy(global_margins)
        for fam, (clf, classes) in self.specialists_.items():
            d = clf.decision_function(X_eval)
            if d.ndim == 1:
                d = np.column_stack([-d, d])
            max_d = np.max(d, axis=1, keepdims=True)
            exp_d = np.exp(d - max_d)
            p_cond = exp_d / np.maximum(np.sum(exp_d, axis=1, keepdims=True), 1e-12)
            log_p = np.log(p_cond + 1e-12)
            log_p_centered = log_p - log_p.mean(axis=1, keepdims=True)

            for col_idx, c_name in enumerate(classes):
                if c_name in LABEL_TO_IDX:
                    glob_idx = LABEL_TO_IDX[c_name]
                    refined[:, glob_idx] += self.beta_refine * log_p_centered[:, col_idx]
        return refined


def test_clique_specialist():
    print("Testing IntraFamilyCliqueSpecialist...")
    np.random.seed(42)
    N_tr = 1000
    N_va = 200
    D = 100

    y_tr = np.random.choice(LABELS, size=N_tr)
    y_va = np.random.choice(LABELS, size=N_va)
    X_tr = sp.csr_matrix(np.random.randn(N_tr, D))
    X_va = sp.csr_matrix(np.random.randn(N_va, D))

    spec = IntraFamilyCliqueSpecialist(alpha_ridge=1.0, beta_refine=0.25)
    spec.fit(X_tr, y_tr)
    assert len(spec.specialists_) >= 3, f"Expected at least 3 specialists, got {len(spec.specialists_)}"
    print(f"  Fitted {len(spec.specialists_)} family specialists: {list(spec.specialists_.keys())} [OK]")

    base_margins = np.random.randn(N_va, NUM_CLASSES)
    refined = spec.refine_margins(base_margins, X_va)
    assert refined.shape == base_margins.shape
    assert np.all(np.isfinite(refined))
    assert not np.array_equal(refined, base_margins), "Margins were not modified"
    print("  Refine margins completed with finite outputs [OK]")


if __name__ == "__main__":
    test_clique_specialist()

