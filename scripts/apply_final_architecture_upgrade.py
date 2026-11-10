from pathlib import Path

def apply_upgrades():
    path = Path("notebooks/kaggle_tri_ensemble.py")
    code = path.read_text(encoding="utf-8")

    # 1. Update DualLinearClassifier defaults to measured optimum C=1.5, alpha=1.0, svc_weight=0.70
    old_dual_init = """    def __init__(self, c_svc=0.4, alpha_ridge=4.0, svc_weight=0.65, seed=42, canonical_labels=None):
        self.c_svc = c_svc
        self.alpha_ridge = alpha_ridge
        self.svc_weight = svc_weight"""
    new_dual_init = """    def __init__(self, c_svc=1.5, alpha_ridge=1.0, svc_weight=0.70, seed=42, canonical_labels=None):
        self.c_svc = c_svc
        self.alpha_ridge = alpha_ridge
        self.svc_weight = svc_weight"""
    assert old_dual_init in code, "Target 1 not found"
    code = code.replace(old_dual_init, new_dual_init, 1)

    # 2. Update quick baseline DualLinear call
    old_quick = "quick_clf = DualLinearClassifier(c_svc=0.4, alpha_ridge=4.0, svc_weight=0.65, seed=42, canonical_labels=LABELS)"
    new_quick = "quick_clf = DualLinearClassifier(c_svc=1.5, alpha_ridge=1.0, svc_weight=0.70, seed=42, canonical_labels=LABELS)"
    assert old_quick in code, "Target 2 not found"
    code = code.replace(old_quick, new_quick, 1)

    # 3. Update CV loop DualLinear call
    old_cv_call = "cpu_clf = DualLinearClassifier(c_svc=0.4, alpha_ridge=4.0, svc_weight=0.65, seed=42 + fold, canonical_labels=LABELS)"
    new_cv_call = "cpu_clf = DualLinearClassifier(c_svc=1.5, alpha_ridge=1.0, svc_weight=0.70, seed=42 + fold, canonical_labels=LABELS)"
    assert old_cv_call in code, "Target 3 not found"
    code = code.replace(old_cv_call, new_cv_call, 1)

    # 4. Precompute V_dense_all upfront before fold loop to avoid 15x regex re-parsing
    old_dense_loop = """                # Dense stylometric calibrated stream
                V_dense_tr = RichStyleFeatures().fit_transform([raw_texts_tr[i] for i in tr_idx])
                V_dense_va = RichStyleFeatures().fit_transform([raw_texts_tr[i] for i in va_idx])
                V_dense_te = RichStyleFeatures().fit_transform(texts_test)
                dense_clf = DenseStylometricClassifier(c=0.2, seed=42 + fold, canonical_labels=LABELS)"""

    new_dense_loop = """                # Dense stylometric calibrated stream (sliced from upfront precomputed matrix)
                V_dense_tr = V_dense_all[tr_idx]
                V_dense_va = V_dense_all[va_idx]
                V_dense_te = V_dense_all[n_tr:]
                dense_clf = DenseStylometricClassifier(c=0.2, seed=42 + fold, canonical_labels=LABELS)"""
    assert old_dense_loop in code, "Target 4 not found"
    code = code.replace(old_dense_loop, new_dense_loop, 1)

    # 5. Insert V_dense_all precomputation right before fold loop
    old_pre_cv = """    # --- 5-Fold CV loop ---
    print("=" * 65, flush=True)"""
    new_pre_cv = """    # Precompute dense stylometrics once upfront (eliminates 15 redundant regex passes across folds)
    t_style_start = time.perf_counter()
    print("  Precomputing dense stylometric features upfront across all passages...", flush=True)
    V_dense_all = RichStyleFeatures().fit_transform(raw_texts_tr + texts_test)
    n_tr = len(raw_texts_tr)
    print(f"  Dense stylometrics matrix shape: {V_dense_all.shape} computed in {time.perf_counter() - t_style_start:.2f}s [OK]\\n", flush=True)

    # --- 5-Fold CV loop ---
    print("=" * 65, flush=True)"""
    assert old_pre_cv in code, "Target 5 not found"
    code = code.replace(old_pre_cv, new_pre_cv, 1)

    path.write_text(code, encoding="utf-8")
    print(f"Successfully applied upgrades to {path}")

if __name__ == "__main__":
    apply_upgrades()

