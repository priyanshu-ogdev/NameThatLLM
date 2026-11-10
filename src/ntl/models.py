"""Fold-local representations, preserving case, punctuation, and line breaks."""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

BASELINES = ("word_lr", "char_lr", "union_lr", "union_svm", "style_svm")
LINEAR_GRID = {f"multi_c{round(c * 100):03d}_s{round(s * 100):03d}": (c, s)
               for c in (0.15, 0.5, 1.5) for s in (0.05, 0.2, 0.5)}
CANDIDATES = BASELINES + ("wide_svm", "punct_svm", "multiview_svm", "wide_rbf", "style_rbf", "style_hgb", "stacked", "topic_svm", "cosine_svm") + tuple(LINEAR_GRID)


class StyleFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            n = max(len(text), 1)
            words = text.split()
            rows.append([
                np.log1p(len(text)), np.log1p(len(words)),
                len(text) / max(len(words), 1),
                sum(c.isupper() for c in text) / n,
                sum(c.isdigit() for c in text) / n,
                *[text.count(p) / n for p in [".", ",", ";", ":", "!", "?", "—", "-", "\n", "\"", "'", "(", "#", "*"]],
                len(re.findall(r"\b\w+['’]\w+\b", text)) / max(len(words), 1),
            ])
        return np.asarray(rows, dtype=np.float64)


class RichStyleFeatures(TransformerMixin, BaseEstimator):
    """Counts, rates, and length summaries; no fitted vocabulary or prompt labels."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        punctuation = ['—', '–', '-', ',', '.', ';', ':', '!', '?', '…', '...',
                       '"', "'", '’', '“', '”', '(', ')', '[', ']', '*', '#', '\n', '\t', '/', '%']
        for text in X:
            words = re.findall(r"\b\w+(?:['’]\w+)?\b", text)
            lower = [word.lower() for word in words]
            lengths = np.array([len(word) for word in words] or [0])
            sentences = [s for s in re.split(r'[.!?]+\s*', text) if s.strip()]
            sentence_lengths = np.array([len(s.split()) for s in sentences] or [0])
            nw, nc = max(len(words), 1), max(len(text), 1)
            counts = np.array([text.count(p) for p in punctuation], dtype=float)
            function_words = 'the a an and or but if then so because although however therefore thus moreover furthermore indeed perhaps might could would should may must can will we you i it they this that these those not no very more most such'.split()
            rows.append([
                np.log1p(nc), np.log1p(nw), np.log1p(len(sentences)),
                float(lengths.mean()), float(lengths.std()), float(np.median(lengths)),
                float(sentence_lengths.mean()), float(sentence_lengths.std()),
                len(set(lower)) / nw, sum(len(w) <= 3 for w in words) / nw,
                sum(len(w) >= 8 for w in words) / nw,
                sum(c.isupper() for c in text) / nc,
                sum(c.isdigit() for c in text) / nc,
                sum(ord(c) > 127 for c in text) / nc,
                len(re.findall(r"\b\w+['’]\w+\b", text)) / nw,
                *np.log1p(counts), *(counts / nw),
                *[lower.count(word) / nw for word in function_words],
                # Positional Slicing (Openings & Closings)
                float(bool(re.search(r"^\s*#{1,6}\s", text))),
                float(bool(re.search(r"^\s*[-*•]\s|^\s*\d+\.\s", text))),
                float(bool(re.search(r"^\s*\*\*", text))),
                float(bool(re.search(r"^(certainly|sure|here is|as an|to answer|i would)", " ".join(words[:max(nw // 4, 1)]), re.I))),
                float(bool(re.search(r"(overall|in conclusion|in summary|ultimately|hope this helps|to summarize)", " ".join(words[-max(nw // 4, 1):]), re.I))),
                float(text.endswith("?")),
                float(text.endswith("!")),
            ])
        return np.asarray(rows, dtype=np.float64)


def advanced_features(name):
    """Initial feature families declared before evaluating their outer folds."""
    style = Pipeline([("extract", RichStyleFeatures()), ("scale", StandardScaler())])
    if name in ("style_rbf", "style_hgb"):
        return style
    char = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                          sublinear_tf=True, min_df=2, max_features=250_000)
    word = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r"\S+",
                          sublinear_tf=True, min_df=2, max_features=150_000)
    if name == "punct_svm":
        char.set_params(analyzer="char_wb", max_features=200_000)
        word.set_params(token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    branches = [("char", char), ("word", word), ("style", style)]
    if name == "multiview_svm":
        char.set_params(max_features=180_000)
        branches.extend([
            ("boundary", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                  sublinear_tf=True, max_features=100_000)),
            ("punct", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2),
                  lowercase=False, sublinear_tf=True, min_df=2, max_features=100_000)),
        ])
    return FeatureUnion(branches, transformer_weights={"style": 0.2}, n_jobs=1)


def build_model(name, seed=42):
    if name not in CANDIDATES:
        raise ValueError(f"Unknown candidate {name}")
    if name in LINEAR_GRID:
        c, weight = LINEAR_GRID[name]
        features = advanced_features("multiview_svm")
        features.set_params(transformer_weights={"style": weight})
        return Pipeline([("features", features), ("classifier", LinearSVC(C=c,
            class_weight="balanced", dual=True, max_iter=15000, random_state=seed))])
    if name == "stacked":
        from ntl.ensemble import PromptStackClassifier
        return PromptStackClassifier(seed=seed)
    if name in ("topic_svm", "cosine_svm"):
        from ntl.kernels import PromptResidualSVC
        return PromptResidualSVC(rank=24 if name == "topic_svm" else 0, seed=seed)
    if name not in BASELINES:
        features = advanced_features(name)
        if name == "wide_rbf":
            from ntl.kernels import FastRbfSVC
            return Pipeline([("features", features), ("classifier", FastRbfSVC())])
        if name == "style_hgb":
            return Pipeline([("features", features), ("classifier", HistGradientBoostingClassifier(
                max_iter=180, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=5.0, min_samples_leaf=15, early_stopping=False,
                class_weight="balanced", random_state=seed))])
        classifier = (SVC(C=10.0, gamma="scale", class_weight="balanced") if name in ("style_rbf", "wide_rbf") else
                      LinearSVC(C=0.5 if name in ("punct_svm", "multiview_svm") else 1.0,
                                class_weight="balanced", dual=True, max_iter=15000, random_state=seed))
        return Pipeline([("features", features), ("classifier", classifier)])
    word = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, strip_accents=None,
                          sublinear_tf=True, min_df=2, max_features=100_000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                          sublinear_tf=True, min_df=2, max_features=150_000)
    if name == "word_lr":
        features = word
    elif name == "char_lr":
        features = char
    else:
        branches = [("word", word), ("char", char)]
        if name == "style_svm":
            branches.append(("style", Pipeline([("extract", StyleFeatures()), ("scale", StandardScaler())])))
        features = FeatureUnion(branches, n_jobs=1)
    classifier = (LogisticRegression(C=4.0, class_weight="balanced", max_iter=2500, random_state=seed)
                  if name.endswith("lr") else
                  LinearSVC(C=1.0, class_weight="balanced", dual=False if name == "style_svm" else "auto",
                            max_iter=10000, random_state=seed))
    return Pipeline([("features", features), ("classifier", classifier)])
