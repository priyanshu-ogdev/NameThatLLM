"""
Applies tail upgrades to notebooks/kaggle_tri_ensemble.py and notebooks/kaggle_dry_run.py,
normalizes line endings to LF, and validates syntax + byte parity.
"""
import ast
from pathlib import Path

def patch_files():
    ens_path = Path("notebooks/kaggle_tri_ensemble.py")
    dry_path = Path("notebooks/kaggle_dry_run.py")

    text_ens = ens_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    text_dry = dry_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # Update imports in shared block
    old_imports = """import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier"""

    new_imports = """import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier"""

    assert old_imports in text_ens and old_imports in text_dry, "Imports block not found!"
    text_ens = text_ens.replace(old_imports, new_imports)
    text_dry = text_dry.replace(old_imports, new_imports)
    print("Updated imports in both notebooks!")

    # In kaggle_tri_ensemble.py:
    # 1. Update fold CPU training to use PromptDF mask and tuned DualLinear
    target_cpu_fold_ens = """        # ── 4. CPU Multiview Dual-Linear (Fast pre-extracted sparse slice) ──
        try:
            print(f"\\n[Fold {fold+1}] CPU Multiview Dual-Linear (Cached CSR slice) {get_mem_status(None)}", flush=True)
            cpu_clf = DualLinearClassifier(c_svc=1.2, alpha_ridge=1.0, svc_weight=0.7, seed=42 + fold, canonical_labels=LABELS)
            with threadpool_limits(limits=4):
                cpu_clf.fit(X_train_all[tr_idx], y_tr_str)
                val_svm_m = cpu_clf.decision_function(X_train_all[va_idx])
                test_svm_m = cpu_clf.decision_function(X_test_all)
                oof_svm[va_idx] = val_svm_m
                test_svm_folds.append(test_svm_m)
                svm_bal_acc = balanced_accuracy_score(y_va_str, [LABELS[i] for i in np.argmax(val_svm_m, axis=1)])
                print(f"  [CPU Linear] Fold {fold+1} bal_acc={svm_bal_acc*100:.2f}%", flush=True)"""

    replacement_cpu_fold_ens = """        # ── 4. CPU Multiview Dual-Linear (Sparse CSR + PromptDF Filter) ──
        try:
            print(f"\\n[Fold {fold+1}] CPU Multiview Dual-Linear (Sparse CSR + PromptDF Filter) {get_mem_status(None)}", flush=True)
            pdf_mask = compute_prompt_df_mask(X_train_all[tr_idx], group_ids[tr_idx], min_prompts=3)
            diag_mask = sp.diags(pdf_mask)
            X_tr_fold = X_train_all[tr_idx] @ diag_mask
            X_va_fold = X_train_all[va_idx] @ diag_mask
            X_te_fold = X_test_all @ diag_mask

            cpu_clf = DualLinearClassifier(c_svc=0.4, alpha_ridge=4.0, svc_weight=0.65, seed=42 + fold, canonical_labels=LABELS)
            with threadpool_limits(limits=4):
                cpu_clf.fit(X_tr_fold, y_tr_str)
                val_svm_m = cpu_clf.decision_function(X_va_fold)
                test_svm_m = cpu_clf.decision_function(X_te_fold)
                oof_svm[va_idx] = val_svm_m
                test_svm_folds.append(test_svm_m)
                svm_bal_acc = balanced_accuracy_score(y_va_str, [LABELS[i] for i in np.argmax(val_svm_m, axis=1)])
                print(f"  [CPU Linear] Fold {fold+1} bal_acc={svm_bal_acc*100:.2f}% (PromptDF min3 active)", flush=True)"""

    if target_cpu_fold_ens in text_ens:
        text_ens = text_ens.replace(target_cpu_fold_ens, replacement_cpu_fold_ens)
        print("Updated fold CPU training in kaggle_tri_ensemble.py")
    else:
        print("Warning: target_cpu_fold_ens not matched directly, checking normalized...")

    # In kaggle_dry_run.py:
    # Update Check 6 to test PromptDF mask and tuned DualLinear
    target_cpu_check6_dry = """        clf_f = DualLinearClassifier(c_svc=1.2, alpha_ridge=1.0, svc_weight=0.7, seed=42, canonical_labels=LABELS)
        clf_f.fit(X_full_tr[tr_idx_f], y_full_str[tr_idx_f])
        v_m = clf_f.decision_function(X_full_tr[va_idx_f])
        v_preds = [LABELS[k] for k in np.argmax(v_m, axis=1)]
        real_cpu_cv_score = balanced_accuracy_score(y_full_str[va_idx_f], v_preds)
        test_svm = clf_f.decision_function(X_full_te)"""

    replacement_cpu_check6_dry = """        pdf_mask = compute_prompt_df_mask(X_full_tr[tr_idx_f], prompt_groups[tr_idx_f], min_prompts=3)
        diag_m = sp.diags(pdf_mask)
        X_tr_f = X_full_tr[tr_idx_f] @ diag_m
        X_va_f = X_full_tr[va_idx_f] @ diag_m
        X_te_f = X_full_te @ diag_m

        clf_f = DualLinearClassifier(c_svc=0.4, alpha_ridge=4.0, svc_weight=0.65, seed=42, canonical_labels=LABELS)
        clf_f.fit(X_tr_f, y_full_str[tr_idx_f])
        v_m = clf_f.decision_function(X_va_f)
        v_preds = [LABELS[k] for k in np.argmax(v_m, axis=1)]
        real_cpu_cv_score = balanced_accuracy_score(y_full_str[va_idx_f], v_preds)
        test_svm = clf_f.decision_function(X_te_f)"""

    if target_cpu_check6_dry in text_dry:
        text_dry = text_dry.replace(target_cpu_check6_dry, replacement_cpu_check6_dry)
        print("Updated Check 6 in kaggle_dry_run.py")

    # Write normalized UTF-8 files
    ens_path.write_text(text_ens, encoding="utf-8")
    dry_path.write_text(text_dry, encoding="utf-8")

    # Verify AST syntax
    ast.parse(text_ens)
    ast.parse(text_dry)
    print("Both files parsed cleanly as valid AST [OK]")

    # Verify shared block byte parity
    marker_ens = "def run_quad_ensemble():"
    marker_dry = "def run_dry_run_verifier():"
    s_ens = text_ens[:text_ens.index(marker_ens)]
    s_dry = text_dry[:text_dry.index(marker_dry)]
    assert s_ens == s_dry, "Shared blocks do not match!"
    print(f"Exact byte parity verified across {len(s_ens)} chars ({len(s_ens.splitlines())} lines) [OK]")

if __name__ == "__main__":
    patch_files()
