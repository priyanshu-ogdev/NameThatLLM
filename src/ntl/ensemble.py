"""Stacking with inner prompt folds; outer evaluation labels never train a blender."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def aligned_scores(model, texts, labels):
    score = model.decision_function(texts) if hasattr(model, "decision_function") else model.predict_proba(texts)
    return score[:, [list(model.classes_).index(label) for label in labels]]


def fit_grouped(model, texts, labels, prompt_ids):
    if getattr(model, "needs_prompt_groups", False):
        return model.fit(texts, labels, prompt_ids=prompt_ids)
    return model.fit(texts, labels)


class PromptStackClassifier(ClassifierMixin, BaseEstimator):
    """Fixed base families and meta regularization, with three inner group folds.

    Every meta-training row is predicted by base models that excluded its prompt.
    During outer evaluation the entire procedure is confined to outer training.
    At inference the base models and meta-classifier use only the passage.
    """
    needs_prompt_groups = True

    def __init__(self, base_names=("wide_svm", "punct_svm", "style_hgb"), inner_splits=3, meta_c=0.1, seed=42):
        self.base_names = base_names
        self.inner_splits = inner_splits
        self.meta_c = meta_c
        self.seed = seed

    def fit(self, X, y, prompt_ids=None):
        from ntl.models import build_model
        if prompt_ids is None:
            raise ValueError("Stacking requires prompt groups for inner validation")
        texts, y, group = np.asarray(X), np.asarray(y), np.asarray(prompt_ids)
        self.classes_ = np.unique(y)
        if len(set(group)) < self.inner_splits:
            raise ValueError("Too few prompt groups for stacking")
        self.inner_fold_ = np.full(len(y), -1, dtype=int)
        splitter = StratifiedGroupKFold(self.inner_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros((len(y), len(self.classes_) * len(self.base_names)))
        for fold, (tr, va) in enumerate(splitter.split(texts, y, group)):
            if set(y[tr]) != set(self.classes_) or set(y[va]) != set(self.classes_):
                raise ValueError("Every inner fold must contain every label")
            if set(group[tr]) & set(group[va]):
                raise ValueError("Inner prompt leakage")
            self.inner_fold_[va] = fold
            for i, name in enumerate(self.base_names):
                model = build_model(name, self.seed)
                model.fit(texts[tr], y[tr])
                start = i * len(self.classes_)
                oof[va, start:start + len(self.classes_)] = aligned_scores(model, texts[va], self.classes_)
        self.meta_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.meta_c,
            class_weight="balanced", max_iter=2500, random_state=self.seed))
        self.meta_.fit(oof, y)
        self.models_ = [build_model(name, self.seed).fit(texts, y) for name in self.base_names]
        return self

    def decision_function(self, X):
        scores = np.hstack([aligned_scores(model, X, self.classes_) for model in self.models_])
        return self.meta_.decision_function(scores)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def rank_blend(
    margins_list,
    weights=None,
    tie_break_margins=None
):
    """
    Zero-parameter within-passage equal-rank summation across arbitrary models.
    Rank_Total = sum_m ( w_m * Rank(margins_m) ) + 1e-6 * normalized_tie_breaker
    """
    from scipy.stats import rankdata

    if not margins_list:
        raise ValueError("margins_list must be non-empty")

    if weights is None:
        weights = [1.0] * len(margins_list)

    if len(weights) != len(margins_list):
        raise ValueError("weights length must match margins_list length")

    total_ranks = np.zeros_like(margins_list[0], dtype=np.float64)
    for m, w in zip(margins_list, weights):
        if m is not None:
            total_ranks += w * np.apply_along_axis(rankdata, 1, m)

    if tie_break_margins is not None:
        ptp = np.ptp(tie_break_margins, axis=1, keepdims=True)
        norm_tie = (tie_break_margins - tie_break_margins.min(axis=1, keepdims=True)) / np.maximum(ptp, 1e-12)
        total_ranks += 1e-6 * norm_tie

    return np.argmax(total_ranks, axis=1)


def sinkhorn_knopp_balance(
    scores: np.ndarray,
    temperature: float = 1.0,
    max_iters: int = 50,
    tol: float = 1e-3
) -> np.ndarray:
    """
    Sinkhorn-Knopp optimal transport balancing on prediction score matrix (N, K).
    Enforces uniform marginal distribution across all K classes on the test set,
    directly maximizing macro Balanced Accuracy and preventing dominant-class collapse.
    """
    N, K = scores.shape
    scaled_scores = scores / max(temperature, 1e-4)
    P = np.exp(scaled_scores - np.max(scaled_scores, axis=1, keepdims=True))
    target_col = np.full(K, N / K, dtype=np.float64)

    r = np.ones(N, dtype=np.float64)
    c = np.ones(K, dtype=np.float64)

    for _ in range(max_iters):
        c_prev = c.copy()
        c = target_col / (np.dot(r, P) + 1e-12)
        r = 1.0 / (np.dot(P, c) + 1e-12)
        if np.max(np.abs(c - c_prev)) < tol:
            break

    P_balanced = r[:, None] * P * c[None, :]
    return P_balanced



def calibrate_thresholds(
    oof_margins: np.ndarray,
    y_true: np.ndarray,
    classes: list,
    step: float = 0.05,
    max_iters: int = 100
) -> np.ndarray:
    """
    Iterative coordinate ascent for marginal threshold offsets to maximize Balanced Accuracy.
    Returns array of additive offsets b of shape (num_classes,).
    """
    from sklearn.metrics import balanced_accuracy_score

    y_indices = np.array([classes.index(label) for label in y_true])
    num_classes = len(classes)
    offsets = np.zeros(num_classes, dtype=np.float64)

    best_score = balanced_accuracy_score(y_indices, np.argmax(oof_margins, axis=1))

    for _ in range(max_iters):
        improved = False
        for c in range(num_classes):
            for delta in (-step, step):
                candidate_offsets = offsets.copy()
                candidate_offsets[c] += delta
                preds = np.argmax(oof_margins + candidate_offsets, axis=1)
                score = balanced_accuracy_score(y_indices, preds)
                if score > best_score:
                    best_score = score
                    offsets = candidate_offsets
                    improved = True
        if not improved:
            break

    return offsets

