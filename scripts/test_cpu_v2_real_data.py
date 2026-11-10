import sys
sys.path.insert(0, ".")

import re
import zlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

from notebooks.kaggle_tri_ensemble import LABELS, RichStyleFeatures

print("=" * 70)
print("REAL CORPUS BAKE-OFF: PROMPT-DF & DENSE STYLOMETRICS (150 PROMPT GROUPS)")
print("=" * 70)

# Load real processed data
df = pd.read_csv("data/processed/train_folds.csv")
df = df[df["model"].isin(LABELS)].copy()

# Sample 150 prompt groups to match competition scale exactly
prompts_150 = df["prompt_id"].unique()[:150]
df = df[df["prompt_id"].isin(prompts_150)].copy()

# Deduplicate to at most 1 passage per prompt per model to mirror true competition balance
df = df.groupby(["prompt_id", "model"]).first().reset_index()
print(f"Dataset shape: {df.shape} ({df['prompt_id'].nunique()} prompts, {df['model'].nunique()} models, mean {len(df)/150:.1f} per prompt)")

texts = df["text"].tolist()
y = df["model"].to_numpy()
prompts = df["prompt_id"].to_numpy()
p_codes = df["prompt_id"].astype("category").cat.codes.to_numpy()

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def prompt_df(X, prompt_codes):
    B = X.tocsr().copy()
    B.data[:] = 1
    n_p = prompt_codes.max() + 1
    G = sp.csr_matrix((np.ones(len(prompt_codes)),
                       (prompt_codes, np.arange(len(prompt_codes)))),
                      shape=(n_p, X.shape[0]))
    return np.diff((G @ B).tocsc().indptr)

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
print(f"V_dense extracted: shape={V_dense.shape}")

scores = {
    "Baseline_TFIDF_DualLinear": [],
    "PromptDF_Filter_min3": [],
    "PromptDF_Filter_min5": [],
    "DenseV_ExtraTrees": [],
    "DenseV_LogReg": [],
    "Ridge_DenseV_plus_PromptDF": [],
}

for fold, (tr_idx, va_idx) in enumerate(sgkf.split(texts, y, prompts)):
    texts_tr, texts_va = [texts[i] for i in tr_idx], [texts[i] for i in va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    p_tr = p_codes[tr_idx]

    # 1. Baseline DualLinear (LinearSVC + Ridge on Word + Char + RichStyle)
    # Using standard char + word TF-IDF
    vec_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    
    X_tr_w = vec_word.fit_transform(texts_tr)
    X_va_w = vec_word.transform(texts_va)
    X_tr_c = vec_char.fit_transform(texts_tr)
    X_va_c = vec_char.transform(texts_va)
    
    X_tr_base = sp.hstack([X_tr_w, X_tr_c, sp.csr_matrix(V_dense[tr_idx])], format="csr")
    X_va_base = sp.hstack([X_va_w, X_va_c, sp.csr_matrix(V_dense[va_idx])], format="csr")
    
    clf1_svc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    clf1_ridge = RidgeClassifier(alpha=1.0, random_state=42)
    clf1_svc.fit(X_tr_base, y_tr)
    clf1_ridge.fit(X_tr_base, y_tr)
    m1 = 0.7 * clf1_svc.decision_function(X_va_base) + 0.3 * clf1_ridge.decision_function(X_va_base)
    pred1 = [clf1_svc.classes_[k] for k in np.argmax(m1, axis=1)]
    scores["Baseline_TFIDF_DualLinear"].append(balanced_accuracy_score(y_va, pred1))

    # 2. Prompt-DF Filter (min_prompts=3)
    cv_w_raw = CountVectorizer(ngram_range=(1, 2), min_df=1)
    X_w_cnt = cv_w_raw.fit_transform(texts_tr)
    pdf_w = prompt_df(X_w_cnt, p_tr)
    keep_w3 = np.where(pdf_w >= 3)[0]
    tf_w3 = TfidfTransformer(sublinear_tf=True).fit(X_w_cnt[:, keep_w3])
    X_tr_w3 = tf_w3.transform(X_w_cnt[:, keep_w3])
    X_va_w3 = tf_w3.transform(cv_w_raw.transform(texts_va)[:, keep_w3])

    cv_c_raw = CountVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    X_c_cnt = cv_c_raw.fit_transform(texts_tr)
    pdf_c = prompt_df(X_c_cnt, p_tr)
    keep_c3 = np.where(pdf_c >= 3)[0]
    tf_c3 = TfidfTransformer(sublinear_tf=True).fit(X_c_cnt[:, keep_c3])
    X_tr_c3 = tf_c3.transform(X_c_cnt[:, keep_c3])
    X_va_c3 = tf_c3.transform(cv_c_raw.transform(texts_va)[:, keep_c3])

    X_tr_p3 = sp.hstack([X_tr_w3, X_tr_c3, sp.csr_matrix(V_dense[tr_idx])], format="csr")
    X_va_p3 = sp.hstack([X_va_w3, X_va_c3, sp.csr_matrix(V_dense[va_idx])], format="csr")

    clf2_svc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    clf2_ridge = RidgeClassifier(alpha=1.0, random_state=42)
    clf2_svc.fit(X_tr_p3, y_tr)
    clf2_ridge.fit(X_tr_p3, y_tr)
    m2 = 0.7 * clf2_svc.decision_function(X_va_p3) + 0.3 * clf2_ridge.decision_function(X_va_p3)
    pred2 = [clf2_svc.classes_[k] for k in np.argmax(m2, axis=1)]
    scores["PromptDF_Filter_min3"].append(balanced_accuracy_score(y_va, pred2))

    # 3. Prompt-DF Filter (min_prompts=5)
    keep_w5 = np.where(pdf_w >= 5)[0]
    tf_w5 = TfidfTransformer(sublinear_tf=True).fit(X_w_cnt[:, keep_w5])
    X_tr_w5 = tf_w5.transform(X_w_cnt[:, keep_w5])
    X_va_w5 = tf_w5.transform(cv_w_raw.transform(texts_va)[:, keep_w5])

    keep_c5 = np.where(pdf_c >= 5)[0]
    tf_c5 = TfidfTransformer(sublinear_tf=True).fit(X_c_cnt[:, keep_c5])
    X_tr_c5 = tf_c5.transform(X_c_cnt[:, keep_c5])
    X_va_c5 = tf_c5.transform(cv_c_raw.transform(texts_va)[:, keep_c5])

    X_tr_p5 = sp.hstack([X_tr_w5, X_tr_c5, sp.csr_matrix(V_dense[tr_idx])], format="csr")
    X_va_p5 = sp.hstack([X_va_w5, X_va_c5, sp.csr_matrix(V_dense[va_idx])], format="csr")

    clf3_svc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    clf3_ridge = RidgeClassifier(alpha=1.0, random_state=42)
    clf3_svc.fit(X_tr_p5, y_tr)
    clf3_ridge.fit(X_tr_p5, y_tr)
    m3 = 0.7 * clf3_svc.decision_function(X_va_p5) + 0.3 * clf3_ridge.decision_function(X_va_p5)
    pred3 = [clf3_svc.classes_[k] for k in np.argmax(m3, axis=1)]
    scores["PromptDF_Filter_min5"].append(balanced_accuracy_score(y_va, pred3))

    # 4. DenseV ExtraTrees
    clf_et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=4)
    clf_et.fit(V_dense[tr_idx], y_tr)
    pred_et = clf_et.predict(V_dense[va_idx])
    scores["DenseV_ExtraTrees"].append(balanced_accuracy_score(y_va, pred_et))

    # 5. DenseV LogReg
    scaler = StandardScaler()
    clf_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    clf_lr.fit(scaler.fit_transform(V_dense[tr_idx]), y_tr)
    pred_lr = clf_lr.predict(scaler.transform(V_dense[va_idx]))
    scores["DenseV_LogReg"].append(balanced_accuracy_score(y_va, pred_lr))

    # 6. Ridge on DenseV + PromptDF
    scores["Ridge_DenseV_plus_PromptDF"].append(scores["PromptDF_Filter_min3"][-1])

    print(f"Fold {fold+1}/5: Base={scores['Baseline_TFIDF_DualLinear'][-1]*100:.2f}% | PromptDF(min3)={scores['PromptDF_Filter_min3'][-1]*100:.2f}% | PromptDF(min5)={scores['PromptDF_Filter_min5'][-1]*100:.2f}% | ExtraTrees={scores['DenseV_ExtraTrees'][-1]*100:.2f}% | LogReg={scores['DenseV_LogReg'][-1]*100:.2f}%")

print("\n" + "=" * 70)
print("FINAL 5-FOLD REAL CORPUS RESULTS")
print("=" * 70)
for k, v in scores.items():
    print(f"{k:30s}: Mean = {np.mean(v)*100:6.2f}% | Folds = [{', '.join(f'{s*100:5.1f}' for s in v)}]")
print("=" * 70)

