"""Compute sparse text RBF kernels once rather than inside each binary SVM."""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC


class PromptResidualSVC(ClassifierMixin, BaseEstimator):
    """Remove training prompt-mean directions from a text kernel.

    The small prompt Gram matrix defines the nuisance subspace. Inference uses
    only text similarities to retained training passages; test prompt IDs are not
    needed. Rank zero provides the otherwise identical cosine-kernel control.
    """
    needs_prompt_groups = True

    def __init__(self, rank=24, strength=0.9, C=2.0, seed=42):
        self.rank = rank
        self.strength = strength
        self.C = C
        self.seed = seed

    def fit(self, X, y, prompt_ids=None):
        from ntl.models import advanced_features
        if prompt_ids is None:
            raise ValueError("Prompt residual model requires training prompt IDs")
        if not 0 <= self.strength <= 1:
            raise ValueError("Projection strength must be in [0,1]")
        self.features_ = advanced_features("multiview_svm")
        self.X_fit_ = self.features_.fit_transform(X, y).tocsr()
        raw = (self.X_fit_ @ self.X_fit_.T).toarray()
        unique, group = np.unique(np.asarray(prompt_ids), return_inverse=True)
        averages = np.zeros((len(unique), len(y)))
        averages[group, np.arange(len(y))] = 1 / np.bincount(group)[group]
        centering = np.eye(len(unique)) - np.ones((len(unique), len(unique))) / len(unique)
        means_kernel = centering @ (averages @ raw @ averages.T) @ centering
        eigenvalues, eigenvectors = np.linalg.eigh(means_kernel)
        keep = np.flatnonzero(eigenvalues > 1e-10)[::-1][:self.rank]
        self.projection_ = averages.T @ centering @ (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]))
        self.Z_fit_ = raw @ self.projection_
        residual = raw - self.strength * (self.Z_fit_ @ self.Z_fit_.T)
        self.fit_norm_ = np.sqrt(np.maximum(np.diag(residual), 1e-12))
        kernel = residual / np.outer(self.fit_norm_, self.fit_norm_)
        self.classifier_ = SVC(C=self.C, kernel="precomputed", class_weight="balanced")
        self.classifier_.fit(kernel, y)
        self.classes_ = self.classifier_.classes_
        return self

    def _kernel(self, X):
        matrix = self.features_.transform(X).tocsr()
        raw = (matrix @ self.X_fit_.T).toarray()
        z = raw @ self.projection_
        norm = np.sqrt(np.maximum(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
                                  - self.strength * np.sum(z * z, axis=1), 1e-12))
        return (raw - self.strength * (z @ self.Z_fit_.T)) / np.outer(norm, self.fit_norm_)

    def decision_function(self, X):
        return self.classifier_.decision_function(self._kernel(X))

    def predict(self, X):
        return self.classifier_.predict(self._kernel(X))


class FastRbfSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, C=10.0, class_weight="balanced"):
        self.C = C
        self.class_weight = class_weight

    def fit(self, X, y):
        self.X_fit_ = X.copy()
        if sparse.issparse(X):
            variance = float(X.multiply(X).mean()) - float(X.mean()) ** 2
        else:
            variance = float(X.var())
        self.gamma_ = 1.0 / (X.shape[1] * variance) if variance else 1.0
        gram = rbf_kernel(X, gamma=self.gamma_)
        self.classifier_ = SVC(C=self.C, kernel="precomputed", class_weight=self.class_weight)
        self.classifier_.fit(gram, y)
        self.classes_ = self.classifier_.classes_
        return self

    def decision_function(self, X):
        return self.classifier_.decision_function(rbf_kernel(X, self.X_fit_, gamma=self.gamma_))

    def predict(self, X):
        return self.classifier_.predict(rbf_kernel(X, self.X_fit_, gamma=self.gamma_))
