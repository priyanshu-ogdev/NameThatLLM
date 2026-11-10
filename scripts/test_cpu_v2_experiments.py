import sys
sys.path.insert(0, ".")

import re
import zlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

from notebooks.kaggle_tri_ensemble import LABELS, make_prompt_groups, RichStyleFeatures

print("=" * 70)
print("CPU V2 PIPELINE BENCHMARK & EXPERIMENTAL BAKE-OFF")
print("Testing Prompt-DF (3A), Topic Masking (3B), Dense Style V + Trees (3C)")
print("=" * 70)

# Load training data
train_df = pd.read_csv("data/sample/train.csv")
print(f"Loaded train_df: {train_df.shape}, {train_df['prompt_id'].nunique()} prompts, {train_df['model'].nunique()} models")

texts = train_df["text"].tolist()
y = train_df["model"].to_numpy()
prompts = train_df["prompt_id"].to_numpy()
p_codes = train_df["prompt_id"].astype("category").cat.codes.to_numpy()

# -------------------------------------------------------------
# 1. Benchmark: Baseline DualLinear (LinearSVC + Ridge) on 5-fold SGKF
# -------------------------------------------------------------
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# --- 3A: Prompt-DF Calculation ---
def prompt_df(X, prompt_codes):
    """Number of distinct prompts each feature occurs in."""
    B = X.tocsr().copy()
    B.data[:] = 1
    n_p = prompt_codes.max() + 1
    G = sp.csr_matrix((np.ones(len(prompt_codes)),
                       (prompt_codes, np.arange(len(prompt_codes)))),
                      shape=(n_p, X.shape[0]))
    return np.diff((G @ B).tocsc().indptr)

# --- 3B: Topic Masking ---
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
TOK_RE  = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?|\d+|\s+|[^\w\s]")

def build_keep_vocab(texts_subset, p_codes_subset, min_frac=0.4):
    seen = defaultdict(set)
    for t, p in zip(texts_subset, p_codes_subset):
        for w in {x.lower() for x in WORD_RE.findall(t)}:
            seen[w].add(p)
    n = len(set(p_codes_subset))
    return {w for w, s in seen.items() if len(s) >= max(min_frac * n, 1)}

def mask_text(t, keep):
    out = []
    for m in TOK_RE.finditer(t):
        s = m.group(0)
        if s.isspace() or not s[0].isalnum():
            out.append(s)
        elif s.isdigit():
            out.append("0" * min(len(s), 3))
        elif s.lower() in keep:
            out.append(s)
        else:
            out.append("Xx" if s[0].isupper() else "xx")
    return "".join(out)

# --- 3C: Dense Stylometric Extractor ---
def extract_dense_v(texts_list):
    rich = RichStyleFeatures()
    V_base = rich.transform(texts_list)
    
    extra = []
    for t in texts_list:
        n = max(len(t), 1)
        w = t.split()
        nw = max(len(w), 1)
        L = t.split("\n")
        nl = max(len(L), 1)
        sl = [len(s.split()) for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()] or [0]
        extra.append([
            len(re.findall(r"^# ", t, re.M)),
            len(re.findall(r"^## ", t, re.M)),
            len(re.findall(r"^###", t, re.M)),
            t.count("**") / 2,
            len(re.findall(r"^\s*[-*•] ", t, re.M)),
            len(re.findall(r"^\s*\d+[.)] ", t, re.M)),
            len(re.findall(r"^\s*\|", t, re.M)),
            len(re.findall(r"^\s*(?:---|\*\*\*)\s*$", t, re.M)),
            t.count("```") / 2,
            1000 * t.count("\u2014") / n,
            1000 * t.count("\u2019") / n,
            1000 * t.count("'") / n,
            sum(ord(c) > 0x1F000 for c in t) / n,
            sum(l != l.rstrip() for l in L) / nl,
            t.count("  ") / nw,
            len(zlib.compress(t.encode())) / max(len(t.encode()), 1),
            float(t.rstrip().endswith("?")),
            len(L[-1].split()),
            len(L[0].split()),
            float(np.percentile(sl, 10)),
            float(np.percentile(sl, 50)),
            float(np.percentile(sl, 90)),
        ])
    return np.hstack([V_base, np.asarray(extra, dtype=np.float64)])

V_dense = extract_dense_v(texts)
print(f"Extracted Dense Style Vector V: shape={V_dense.shape}")

# -------------------------------------------------------------
# 2. Run Comparative 5-Fold Cross-Validation Bake-Off
# -------------------------------------------------------------
scores = {
    "Baseline_TFIDF_SVC": [],
    "PromptDF_Filter_SVC": [],
    "TopicMasked_TFIDF_SVC": [],
    "DenseV_LogReg": [],
    "DenseV_ExtraTrees": [],
    "Blend_PromptDF_plus_DenseTrees": [],
}

for fold, (tr_idx, va_idx) in enumerate(sgkf.split(texts, y, prompts)):
    texts_tr, texts_va = [texts[i] for i in tr_idx], [texts[i] for i in va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    p_tr = p_codes[tr_idx]

    # Model 1: Standard TF-IDF + LinearSVC
    cv_base = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_tr_b = cv_base.fit_transform(texts_tr)
    X_va_b = cv_base.transform(texts_va)
    clf1 = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    clf1.fit(X_tr_b, y_tr)
    pred1 = clf1.predict(X_va_b)
    scores["Baseline_TFIDF_SVC"].append(balanced_accuracy_score(y_va, pred1))

    # Model 2: Prompt-DF Filtered TF-IDF (>= 2 prompts)
    cv_cnt = CountVectorizer(ngram_range=(1, 2), min_df=1)
    X_cnt_tr = cv_cnt.fit_transform(texts_tr)
    p_df = prompt_df(X_cnt_tr, p_tr)
    keep_feat = np.where(p_df >= 2)[0]
    if len(keep_feat) == 0:
        keep_feat = np.arange(X_cnt_tr.shape[1])
    tfidf_p = TfidfTransformer(sublinear_tf=True).fit(X_cnt_tr[:, keep_feat])
    X_tr_p = tfidf_p.transform(X_cnt_tr[:, keep_feat])
    X_va_cnt = cv_cnt.transform(texts_va)
    X_va_p = tfidf_p.transform(X_va_cnt[:, keep_feat])
    
    clf2 = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    clf2.fit(X_tr_p, y_tr)
    pred2 = clf2.predict(X_va_p)
    scores["PromptDF_Filter_SVC"].append(balanced_accuracy_score(y_va, pred2))

    # Model 3: Topic Masked N-Grams
    keep_vocab = build_keep_vocab(texts_tr, p_tr, min_frac=0.3)
    masked_tr = [mask_text(t, keep_vocab) for t in texts_tr]
    masked_va = [mask_text(t, keep_vocab) for t in texts_va]
    cv_m = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    X_tr_m = cv_m.fit_transform(masked_tr)
    X_va_m = cv_m.transform(masked_va)
    clf3 = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    clf3.fit(X_tr_m, y_tr)
    pred3 = clf3.predict(X_va_m)
    scores["TopicMasked_TFIDF_SVC"].append(balanced_accuracy_score(y_va, pred3))

    # Model 4: Dense Style Vector V with LogisticRegression
    scaler = StandardScaler()
    V_tr = scaler.fit_transform(V_dense[tr_idx])
    V_va = scaler.transform(V_dense[va_idx])
    clf4 = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    clf4.fit(V_tr, y_tr)
    pred4 = clf4.predict(V_va)
    scores["DenseV_LogReg"].append(balanced_accuracy_score(y_va, pred4))

    # Model 5: Dense Style Vector V with ExtraTrees
    clf5 = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=4)
    clf5.fit(V_dense[tr_idx], y_tr)
    pred5 = clf5.predict(V_dense[va_idx])
    scores["DenseV_ExtraTrees"].append(balanced_accuracy_score(y_va, pred5))

    # Model 6: Blend of PromptDF Linear + Dense Trees
    m2 = clf2.decision_function(X_va_p)
    c_to_idx5 = {c: i for i, c in enumerate(clf5.classes_)}
    p5 = clf5.predict_proba(V_dense[va_idx])
    p5_aligned = np.zeros_like(m2)
    for col, lab in enumerate(clf2.classes_):
        if lab in c_to_idx5:
            p5_aligned[:, col] = p5[:, c_to_idx5[lab]]
    # Soft blend
    m_blend = m2 + 2.0 * np.log(np.clip(p5_aligned, 1e-4, 1.0))
    pred6 = [clf2.classes_[k] for k in np.argmax(m_blend, axis=1)]
    scores["Blend_PromptDF_plus_DenseTrees"].append(balanced_accuracy_score(y_va, pred6))

print("\n" + "=" * 70)
print("5-FOLD GROUPED CROSS-VALIDATION RESULTS (BAKE-OFF)")
print("=" * 70)
for name, sc in scores.items():
    print(f"{name:32s}: Mean = {np.mean(sc)*100:6.2f}% | Folds = [{', '.join(f'{s*100:5.1f}' for s in sc)}]")
print("=" * 70)

