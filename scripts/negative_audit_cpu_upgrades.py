import sys
sys.path.insert(0, ".")

import re
import zlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

from notebooks.kaggle_tri_ensemble import LABELS, RichStyleFeatures

def prompt_df(X, prompt_codes):
    B = X.tocsr().copy()
    B.data[:] = 1
    n_p = prompt_codes.max() + 1
    G = sp.csr_matrix((np.ones(len(prompt_codes)),
                       (prompt_codes, np.arange(len(prompt_codes)))),
                      shape=(n_p, X.shape[0]))
    return np.diff((G @ B).tocsc().indptr)

def main():
    print("=" * 75, flush=True)
    print("NEGATIVE AUDIT: CPU PIPELINE RIGOROUS 5-FOLD BAKE-OFF (N=2,023, 150 PROMPTS)", flush=True)
    print("Acceptance Criteria: Gain >= 1.5% on mean AND win in >= 4/5 folds", flush=True)
    print("=" * 75, flush=True)

    df = pd.read_parquet("data/processed/clean_corpus_long.parquet")
    top_prompts = df.groupby("prompt_id")["model"].nunique().nlargest(150).index
    sub = df[df["prompt_id"].isin(top_prompts)].copy()
    sub = sub.groupby(["prompt_id", "model"]).first().reset_index()

    texts = sub["text"].tolist()
    y = sub["model"].to_numpy()
    prompts = sub["prompt_id"].to_numpy()
    p_codes = sub["prompt_id"].astype("category").cat.codes.to_numpy()

    print(f"Data slice: {len(sub)} passages, {sub['prompt_id'].nunique()} prompts, {sub['model'].nunique()} classes\n", flush=True)

    rich = RichStyleFeatures()
    V_base = rich.transform(texts)
    print(f"V_base extracted: shape={V_base.shape}", flush=True)

    extra = []
    for t in texts:
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
    V_167 = np.hstack([V_base, np.asarray(extra, dtype=np.float64)])
    print(f"V_167 extracted: shape={V_167.shape}\n", flush=True)

    results = {
        "Baseline_145": [],
        "Expanded_167": [],
        "PromptDF_min5": [],
        "PromptDF_min8": [],
        "ExtraTrees_Dense": [],
        "Blend_Linear_ET": [],
    }

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(texts, y, prompts)):
        texts_tr, texts_va = [texts[i] for i in tr_idx], [texts[i] for i in va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        p_tr = p_codes[tr_idx]

        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
        X_tr_w, X_va_w = vw.fit_transform(texts_tr), vw.transform(texts_va)
        X_tr_c, X_va_c = vc.fit_transform(texts_tr), vc.transform(texts_va)

        X_tr_base = sp.hstack([X_tr_w, X_tr_c, sp.csr_matrix(V_base[tr_idx])], format="csr")
        X_va_base = sp.hstack([X_va_w, X_va_c, sp.csr_matrix(V_base[va_idx])], format="csr")

        clf_svc = LinearSVC(C=1.0, max_iter=600, tol=1e-3, dual="auto", random_state=42)
        clf_ridge = RidgeClassifier(alpha=1.0, random_state=42)
        clf_svc.fit(X_tr_base, y_tr)
        clf_ridge.fit(X_tr_base, y_tr)
        m_base = 0.7 * clf_svc.decision_function(X_va_base) + 0.3 * clf_ridge.decision_function(X_va_base)
        pred_base = [clf_svc.classes_[k] for k in np.argmax(m_base, axis=1)]
        s_base = balanced_accuracy_score(y_va, pred_base)
        results["Baseline_145"].append(s_base)

        X_tr_167 = sp.hstack([X_tr_w, X_tr_c, sp.csr_matrix(V_167[tr_idx])], format="csr")
        X_va_167 = sp.hstack([X_va_w, X_va_c, sp.csr_matrix(V_167[va_idx])], format="csr")
        clf_svc.fit(X_tr_167, y_tr)
        clf_ridge.fit(X_tr_167, y_tr)
        m_167 = 0.7 * clf_svc.decision_function(X_va_167) + 0.3 * clf_ridge.decision_function(X_va_167)
        pred_167 = [clf_svc.classes_[k] for k in np.argmax(m_167, axis=1)]
        s_167 = balanced_accuracy_score(y_va, pred_167)
        results["Expanded_167"].append(s_167)

        # PromptDF min5
        cv_w_raw = CountVectorizer(ngram_range=(1, 2), min_df=1)
        X_w_cnt = cv_w_raw.fit_transform(texts_tr)
        pdf_w = prompt_df(X_w_cnt, p_tr)
        keep_w5 = np.where(pdf_w >= 5)[0]
        tf_w5 = TfidfTransformer(sublinear_tf=True).fit(X_w_cnt[:, keep_w5])
        X_tr_w5 = tf_w5.transform(X_w_cnt[:, keep_w5])
        X_va_w5 = tf_w5.transform(cv_w_raw.transform(texts_va)[:, keep_w5])

        X_tr_p5 = sp.hstack([X_tr_w5, X_tr_c, sp.csr_matrix(V_167[tr_idx])], format="csr")
        X_va_p5 = sp.hstack([X_va_w5, X_va_c, sp.csr_matrix(V_167[va_idx])], format="csr")
        clf_svc.fit(X_tr_p5, y_tr)
        clf_ridge.fit(X_tr_p5, y_tr)
        m_p5 = 0.7 * clf_svc.decision_function(X_va_p5) + 0.3 * clf_ridge.decision_function(X_va_p5)
        pred_p5 = [clf_svc.classes_[k] for k in np.argmax(m_p5, axis=1)]
        s_p5 = balanced_accuracy_score(y_va, pred_p5)
        results["PromptDF_min5"].append(s_p5)

        # PromptDF min8
        keep_w8 = np.where(pdf_w >= 8)[0]
        tf_w8 = TfidfTransformer(sublinear_tf=True).fit(X_w_cnt[:, keep_w8])
        X_tr_w8 = tf_w8.transform(X_w_cnt[:, keep_w8])
        X_va_w8 = tf_w8.transform(cv_w_raw.transform(texts_va)[:, keep_w8])

        X_tr_p8 = sp.hstack([X_tr_w8, X_tr_c, sp.csr_matrix(V_167[tr_idx])], format="csr")
        X_va_p8 = sp.hstack([X_va_w8, X_va_c, sp.csr_matrix(V_167[va_idx])], format="csr")
        clf_svc.fit(X_tr_p8, y_tr)
        clf_ridge.fit(X_tr_p8, y_tr)
        m_p8 = 0.7 * clf_svc.decision_function(X_va_p8) + 0.3 * clf_ridge.decision_function(X_va_p8)
        pred_p8 = [clf_svc.classes_[k] for k in np.argmax(m_p8, axis=1)]
        s_p8 = balanced_accuracy_score(y_va, pred_p8)
        results["PromptDF_min8"].append(s_p8)

        # ExtraTrees
        clf_et = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=1)
        clf_et.fit(V_167[tr_idx], y_tr)
        pred_et = clf_et.predict(V_167[va_idx])
        s_et = balanced_accuracy_score(y_va, pred_et)
        results["ExtraTrees_Dense"].append(s_et)

        # Blend
        p_et = clf_et.predict_proba(V_167[va_idx])
        c_to_idx = {c: i for i, c in enumerate(clf_et.classes_)}
        p_et_aligned = np.zeros_like(m_base)
        for col, lab in enumerate(clf_svc.classes_):
            if lab in c_to_idx:
                p_et_aligned[:, col] = p_et[:, c_to_idx[lab]]
        m_blend = m_base + 0.3 * np.log(np.clip(p_et_aligned, 1e-4, 1.0))
        pred_blend = [clf_svc.classes_[k] for k in np.argmax(m_blend, axis=1)]
        s_blend = balanced_accuracy_score(y_va, pred_blend)
        results["Blend_Linear_ET"].append(s_blend)

        print(f"  [Fold {fold+1}/5] Base: {s_base*100:.2f}% | Exp167: {s_167*100:.2f}% | PDF5: {s_p5*100:.2f}% | PDF8: {s_p8*100:.2f}% | ET: {s_et*100:.2f}% | Blend: {s_blend*100:.2f}%", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("5-FOLD EVALUATION SUMMARY", flush=True)
    print("=" * 75, flush=True)
    base_arr = np.array(results["Baseline_145"])
    for name, vals in results.items():
        arr = np.array(vals)
        delta = (arr.mean() - base_arr.mean()) * 100
        wins = np.sum(arr > base_arr)
        ties = np.sum(arr == base_arr)
        losses = np.sum(arr < base_arr)
        accept = "PASS" if delta >= 1.5 and wins >= 4 else "FAIL"
        print(f"{name:18s}: Mean = {arr.mean()*100:6.2f}% (Delta = {delta:+5.2f}%) | Wins = {wins}/5, Ties = {ties}/5, Losses = {losses}/5 -> Verdict: {accept}", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
