"""
Comprehensive implementation of all audit fixes into notebooks/kaggle_tri_ensemble.py,
followed by automated generation of notebooks/kaggle_dry_run.py to ensure 100% byte parity.
"""
import ast
import re
from pathlib import Path

def apply_fixes():
    ens_path = Path("notebooks/kaggle_tri_ensemble.py")
    code = ens_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # 1. Imports at top of file
    if "from transformers import AutoTokenizer" not in code:
        old_trans_block = """try:
    import transformers
    transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
except Exception:
    pass"""
        new_trans_block = """try:
    import transformers
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig
    transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
except Exception:
    pass"""
        assert old_trans_block in code, "Could not find old_trans_block"
        code = code.replace(old_trans_block, new_trans_block, 1)

    # Ensure scipy.sparse and LogisticRegression are imported
    if "import scipy.sparse as sp" not in code:
        old_sci = "import pandas as pd"
        new_sci = "import pandas as pd\nimport scipy.sparse as sp"
        code = code.replace(old_sci, new_sci, 1)

    if "from sklearn.linear_model import LogisticRegression" not in code:
        old_lm = "from sklearn.linear_model import RidgeClassifier"
        new_lm = "from sklearn.linear_model import LogisticRegression, RidgeClassifier"
        code = code.replace(old_lm, new_lm, 1)

    # 2. Fix DenseStylometricClassifier __init__ (Low bug)
    old_dense_def = """class DenseStylometricClassifier(BaseEstimator):
    \"\"\"
    Standardized dense stylometrics classifier.
    Scales continuous stylometric features with StandardScaler and fits an L2-regularized
    LogisticRegression classifier to produce calibrated probabilities without coordinate descent distortion.
    \"\"\"
    def __init__(self, c=0.2, seed=42, canonical_labels=None):
        self.c = c
        self.seed = seed
        self.canonical_labels = canonical_labels
        self.scaler_ = StandardScaler()
        self.clf_ = LogisticRegression(
            C=self.c, max_iter=1000, random_state=self.seed, class_weight=None, solver="lbfgs"
        )

    def fit(self, V, y):
        V_scaled = self.scaler_.fit_transform(V)
        self.clf_.fit(V_scaled, y)"""

    new_dense_def = """class DenseStylometricClassifier(BaseEstimator):
    \"\"\"
    Standardized dense stylometrics classifier.
    Scales continuous stylometric features with StandardScaler and fits an L2-regularized
    LogisticRegression classifier to produce calibrated probabilities without coordinate descent distortion.
    \"\"\"
    def __init__(self, c=0.2, seed=42, canonical_labels=None):
        self.c = c
        self.seed = seed
        self.canonical_labels = canonical_labels

    def fit(self, V, y):
        self.scaler_ = StandardScaler()
        self.clf_ = LogisticRegression(
            C=self.c, max_iter=1000, random_state=self.seed, class_weight=None, solver="lbfgs"
        )
        V_scaled = self.scaler_.fit_transform(V)
        self.clf_.fit(V_scaled, y)"""

    if old_dense_def in code:
        code = code.replace(old_dense_def, new_dense_def, 1)

    # 3. Fix LLRD embedding LR (Bug 6)
    old_emb_lr = 'emb_lr = max(lr_backbone * (layer_decay ** 4), 8e-6)'
    new_emb_lr = 'emb_lr = lr_backbone * (layer_decay ** n_layers)'
    if old_emb_lr in code:
        code = code.replace(old_emb_lr, new_emb_lr, 1)

    # 4. Fix Qwen pooling in evaluate_qwen_split (Bug 2)
    old_qwen_eval_pool = """                mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device)
                sum_hidden = (out.last_hidden_state * mask).sum(dim=1).float()
                pooled = sum_hidden / torch.clamp(mask.sum(dim=1).float(), min=1e-9)"""

    new_qwen_eval_pool = """                m = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
                pooled = (out.last_hidden_state.float() * m).sum(1) / m.sum(1).clamp(min=1e-9)"""

    if old_qwen_eval_pool in code:
        code = code.replace(old_qwen_eval_pool, new_qwen_eval_pool, 1)

    # 5. Fix DeBERTa token initialization (Bug 1: unconditional initialization)
    old_token_init = """    # Ensure model embeddings match tokenizer vocabulary size and initialize newline weights
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb_module = model.get_input_embeddings()
            dot_id = tokenizer.convert_tokens_to_ids(".")
            bar_id = tokenizer.convert_tokens_to_ids("|")
            nl_id = tokenizer.convert_tokens_to_ids("[NL]")
            par_id = tokenizer.convert_tokens_to_ids("[PAR]")
            if dot_id is not None and nl_id is not None and nl_id < emb_module.weight.shape[0]:
                emb_module.weight[nl_id].copy_(emb_module.weight[dot_id])
            if bar_id is not None and par_id is not None and par_id < emb_module.weight.shape[0]:
                emb_module.weight[par_id].copy_(emb_module.weight[bar_id])"""

    new_token_init = """    # Ensure model embeddings match tokenizer vocabulary size and initialize newline weights unconditionally
    emb_module = model.get_input_embeddings()
    if len(tokenizer) > emb_module.weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        emb_module = model.get_input_embeddings()
    with torch.no_grad():
        for new_tok, src_tok in (("[NL]", "."), ("[PAR]", "|")):
            n_id = tokenizer.convert_tokens_to_ids(new_tok)
            s_id = tokenizer.convert_tokens_to_ids(src_tok)
            if n_id is not None and s_id is not None and max(n_id, s_id) < emb_module.weight.shape[0]:
                emb_module.weight[n_id].copy_(emb_module.weight[s_id])"""

    if old_token_init in code:
        code = code.replace(old_token_init, new_token_init, 1)

    # 6. Fix cross-device empty_cache() in train_transformer_fold (Bug 4)
    old_empty_cache_trans = """    if cuda_ok:
        n_gpus_local = torch.cuda.device_count()
        for gi in range(n_gpus_local):
            with torch.cuda.device(gi):
                torch.cuda.empty_cache()"""

    new_empty_cache_trans = """    if cuda_ok and str(dev).startswith("cuda"):
        dev_idx = torch.device(dev).index if torch.device(dev).index is not None else 0
        with torch.cuda.device(dev_idx):
            torch.cuda.empty_cache()"""

    if old_empty_cache_trans in code:
        code = code.replace(old_empty_cache_trans, new_empty_cache_trans, 1)

    # 7. Fix OOM retry limit in train_transformer_fold (Bug 7)
    old_trans_oom = """                except torch.cuda.OutOfMemoryError:
                    print(f"  [{model_name}] CUDA OOM detected at micro-step {micro_step}. Resetting partial grads...", flush=True)
                    try:
                        del logits, loss
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    if not enable_grad_checkpointing:
                        print(f"  [{model_name}] Enabling gradient checkpointing on-the-fly to recover VRAM...", flush=True)
                        try:
                            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        except TypeError:
                            model.gradient_checkpointing_enable()
                        enable_grad_checkpointing = True
                    if cuda_ok:
                        torch.cuda.empty_cache()
                    continue"""

    new_trans_oom = """                except torch.cuda.OutOfMemoryError:
                    n_oom_skips += 1
                    print(f"  [{model_name}] CUDA OOM detected at micro-step {micro_step} (OOM #{n_oom_skips}). Resetting partial grads...", flush=True)
                    if n_oom_skips > 5:
                        raise RuntimeError(f"[{model_name}] Exceeded 5 CUDA OOM errors in fold; aborting to prevent data starvation.")
                    try:
                        del logits, loss
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    if not enable_grad_checkpointing:
                        print(f"  [{model_name}] Enabling gradient checkpointing on-the-fly to recover VRAM...", flush=True)
                        try:
                            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        except TypeError:
                            model.gradient_checkpointing_enable()
                        enable_grad_checkpointing = True
                    if cuda_ok and str(dev).startswith("cuda"):
                        dev_idx = torch.device(dev).index if torch.device(dev).index is not None else 0
                        with torch.cuda.device(dev_idx):
                            torch.cuda.empty_cache()
                    continue"""

    if old_trans_oom in code:
        code = code.replace(old_trans_oom, new_trans_oom, 1)

    old_loop_start = 'n_skipped_micros = 0\n            stop_early = False'
    new_loop_start = 'n_skipped_micros = 0\n            n_oom_skips = 0\n            stop_early = False'
    if old_loop_start in code:
        code = code.replace(old_loop_start, new_loop_start, 1)

    # 8. Fix Qwen pooling & NaN-abort threshold in train_qwen_lora_fold (Bugs 2 & 3)
    old_qwen_pool = """                    mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device)
                    sum_hidden = (out.last_hidden_state * mask).sum(dim=1).float()
                    pooled = sum_hidden / torch.clamp(mask.sum(dim=1).float(), min=1e-9)
                    logits = head(pooled.to(head_dev))
                    loss = criterion(logits.float(), batch_labels) / grad_accum

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  [Qwen LoRA] Warning: NaN loss at micro-step {micro_step}. Skipping.", flush=True)
                        if accumulated_steps > 0:
                            optimizer.zero_grad(set_to_none=True)
                            accumulated_steps = 0
                        continue"""

    new_qwen_pool = """                    m = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
                    pooled = (out.last_hidden_state.float() * m).sum(1) / m.sum(1).clamp(min=1e-9)
                    logits = head(pooled.to(head_dev))
                    loss = criterion(logits.float(), batch_labels) / grad_accum

                    if torch.isnan(loss) or torch.isinf(loss):
                        n_skipped_micros += 1
                        print(f"  [Qwen LoRA] Warning: NaN/Inf loss at micro-step {micro_step}. Skipping ({n_skipped_micros} total skips).", flush=True)
                        if n_skipped_micros > 0.15 * max(n_total_micros * (epoch + 1), 1):
                            raise FloatingPointError(f"[Qwen LoRA] >15% micro-batches ({n_skipped_micros}) produced NaN/Inf loss. Discarding fold.")
                        if accumulated_steps > 0:
                            optimizer.zero_grad(set_to_none=True)
                            accumulated_steps = 0
                        continue"""

    if old_qwen_pool in code:
        code = code.replace(old_qwen_pool, new_qwen_pool, 1)

    old_qwen_epoch_start = 'for epoch in range(epochs):\n        epoch_loss = 0.0'
    new_qwen_epoch_start = 'n_skipped_micros = 0\n    for epoch in range(epochs):\n        epoch_loss = 0.0'
    if old_qwen_epoch_start in code:
        code = code.replace(old_qwen_epoch_start, new_qwen_epoch_start, 1)

    # Fix cross-device empty_cache() in train_qwen_lora_fold
    if old_empty_cache_trans in code:
        new_qwen_empty_cache = """    if cuda_ok:
        for gi in (first_dev, head_dev):
            if str(gi).startswith("cuda"):
                dev_idx = torch.device(gi).index if torch.device(gi).index is not None else 0
                with torch.cuda.device(dev_idx):
                    torch.cuda.empty_cache()"""
        code = code.replace(old_empty_cache_trans, new_qwen_empty_cache, 1)

    # 9. Fix Ridge stacker diagnostic grouping (Bug 9)
    old_ridge_stack = "ridge_preds = cross_val_predict(RidgeClassifier(alpha=1.0), X_oof_stack, y_stack, cv=5)"
    new_ridge_stack = "ridge_preds = cross_val_predict(RidgeClassifier(alpha=1.0), X_oof_stack, y_stack, cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), groups=group_ids_stack)"
    if old_ridge_stack in code:
        code = code.replace(old_ridge_stack, new_ridge_stack, 1)

    # 10. Wire in Step 0 baseline submission with PromptDF
    old_step0_quick = """    quick_clf = DualLinearClassifier(c_svc=0.4, alpha_ridge=4.0, svc_weight=0.65, seed=42, canonical_labels=LABELS)
    with threadpool_limits(limits=4):
        quick_clf.fit(X_train_all, y_true)
        quick_test_m = quick_clf.decision_function(X_test_all)"""

    new_step0_quick = """    pdf_mask_quick = compute_prompt_df_mask(X_train_all, group_ids, min_prompts=3)
    diag_mask_quick = sp.diags(pdf_mask_quick)
    quick_clf = DualLinearClassifier(c_svc=0.4, alpha_ridge=4.0, svc_weight=0.65, seed=42, canonical_labels=LABELS)
    with threadpool_limits(limits=4):
        quick_clf.fit(X_train_all @ diag_mask_quick, y_true)
        quick_test_m = quick_clf.decision_function(X_test_all @ diag_mask_quick)"""

    if old_step0_quick in code:
        code = code.replace(old_step0_quick, new_step0_quick, 1)

    # 11. Wire DenseStylometricClassifier into fold loop
    old_fold_cpu = """            cpu_clf = DualLinearClassifier(c_svc=0.4, alpha_ridge=4.0, svc_weight=0.65, seed=42 + fold, canonical_labels=LABELS)
            with threadpool_limits(limits=4):
                cpu_clf.fit(X_tr_fold, y_tr_str)
                val_svm_m = cpu_clf.decision_function(X_va_fold)
                test_svm_m = cpu_clf.decision_function(X_te_fold)
                oof_svm[va_idx] = val_svm_m
                test_svm_folds.append(test_svm_m)
                svm_bal_acc = balanced_accuracy_score(y_va_str, [LABELS[i] for i in np.argmax(val_svm_m, axis=1)])
                print(f"  [CPU Linear] Fold {fold+1} bal_acc={svm_bal_acc*100:.2f}% (PromptDF min3 active)", flush=True)"""

    new_fold_cpu = """            cpu_clf = DualLinearClassifier(c_svc=0.4, alpha_ridge=4.0, svc_weight=0.65, seed=42 + fold, canonical_labels=LABELS)
            with threadpool_limits(limits=4):
                cpu_clf.fit(X_tr_fold, y_tr_str)
                val_sparse_m = cpu_clf.decision_function(X_va_fold)
                test_sparse_m = cpu_clf.decision_function(X_te_fold)

                # Dense stylometric calibrated stream
                V_dense_tr = RichStyleFeatures().fit_transform([raw_texts_tr[i] for i in tr_idx])
                V_dense_va = RichStyleFeatures().fit_transform([raw_texts_tr[i] for i in va_idx])
                V_dense_te = RichStyleFeatures().fit_transform(texts_test)
                dense_clf = DenseStylometricClassifier(c=0.2, seed=42 + fold, canonical_labels=LABELS)
                dense_clf.fit(V_dense_tr, y_tr_str)
                p_dense_va = dense_clf.predict_proba(V_dense_va)
                p_dense_te = dense_clf.predict_proba(V_dense_te)

                # Calibrated consensus: 85% sparse TF-IDF + 15% scaled dense stylometrics
                p_sparse_va = margins_to_probs(val_sparse_m, temp=0.20)
                p_sparse_te = margins_to_probs(test_sparse_m, temp=0.20)
                p_cpu_va = 0.85 * p_sparse_va + 0.15 * p_dense_va
                p_cpu_te = 0.85 * p_sparse_te + 0.15 * p_dense_te
                val_svm_m = 0.20 * np.log(p_cpu_va + 1e-12)
                test_svm_m = 0.20 * np.log(p_cpu_te + 1e-12)

                check_finite_logits(val_svm_m, f"CPU Linear Fold {fold+1} Val")
                check_finite_logits(test_svm_m, f"CPU Linear Fold {fold+1} Test")

                oof_svm[va_idx] = val_svm_m
                test_svm_folds.append(test_svm_m)
                svm_bal_acc = balanced_accuracy_score(y_va_str, [LABELS[i] for i in np.argmax(val_svm_m, axis=1)])
                print(f"  [CPU Linear] Fold {fold+1} bal_acc={svm_bal_acc*100:.2f}% (PromptDF min3 + Dense Stylometrics active)", flush=True)"""

    if old_fold_cpu in code:
        code = code.replace(old_fold_cpu, new_fold_cpu, 1)

    # 12. Check finite logits for DeBERTa, BGE, and Qwen test predictions
    old_deb_res = """                    val_deb, test_deb = f_deb.result()
                    if val_deb is not None and np.all(np.isfinite(val_deb)):
                        oof_deb[va_idx] = val_deb
                        test_deb_folds.append(test_deb)"""

    new_deb_res = """                    val_deb, test_deb = f_deb.result()
                    if val_deb is not None and np.all(np.isfinite(val_deb)):
                        check_finite_logits(val_deb, f"DeBERTa Fold {fold+1} Val")
                        check_finite_logits(test_deb, f"DeBERTa Fold {fold+1} Test")
                        oof_deb[va_idx] = val_deb
                        test_deb_folds.append(test_deb)"""

    if old_deb_res in code:
        code = code.replace(old_deb_res, new_deb_res, 1)

    old_bge_res = """                    val_bge, test_bge = f_bge.result()
                    if val_bge is not None and np.all(np.isfinite(val_bge)):
                        oof_bge[va_idx] = val_bge
                        test_bge_folds.append(test_bge)"""

    new_bge_res = """                    val_bge, test_bge = f_bge.result()
                    if val_bge is not None and np.all(np.isfinite(val_bge)):
                        check_finite_logits(val_bge, f"BGE Fold {fold+1} Val")
                        check_finite_logits(test_bge, f"BGE Fold {fold+1} Test")
                        oof_bge[va_idx] = val_bge
                        test_bge_folds.append(test_bge)"""

    if old_bge_res in code:
        code = code.replace(old_bge_res, new_bge_res, 1)

    old_qwen_res = """                    val_qwen, test_qwen = train_qwen_lora_fold(
                        texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                        qwen_dir, LABELS,
                        first_dev=qwen_first_dev, head_dev=qwen_head_dev,
                        batch_size=8, grad_accum=2, max_length=352,
                        epochs=N_EPOCHS_QWEN, lr=1.8e-4, lora_r=64, lora_alpha=128,
                        deadline=pipeline_deadline,
                    )
                    if val_qwen is not None and np.all(np.isfinite(val_qwen)):
                        oof_qwen[va_idx] = val_qwen
                        test_qwen_folds.append(test_qwen)"""

    new_qwen_res = """                    val_qwen, test_qwen = train_qwen_lora_fold(
                        texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                        qwen_dir, LABELS,
                        first_dev=qwen_first_dev, head_dev=qwen_head_dev,
                        batch_size=8, grad_accum=2, max_length=352,
                        epochs=N_EPOCHS_QWEN, lr=1.8e-4, lora_r=64, lora_alpha=128,
                        deadline=pipeline_deadline,
                    )
                    if val_qwen is not None and np.all(np.isfinite(val_qwen)):
                        check_finite_logits(val_qwen, f"Qwen Fold {fold+1} Val")
                        check_finite_logits(test_qwen, f"Qwen Fold {fold+1} Test")
                        oof_qwen[va_idx] = val_qwen
                        test_qwen_folds.append(test_qwen)"""

    if old_qwen_res in code:
        code = code.replace(old_qwen_res, new_qwen_res, 1)

    # Write patched kaggle_tri_ensemble.py
    ens_path.write_text(code, encoding="utf-8")
    print(f"Successfully patched notebooks/kaggle_tri_ensemble.py ({len(code)} chars, {len(code.splitlines())} lines)")

    # Validate AST
    ast.parse(code)
    print("notebooks/kaggle_tri_ensemble.py parsed cleanly as valid AST [OK]")

    # 13. Rebuild notebooks/kaggle_dry_run.py from updated shared code
    import sys
    sys.path.insert(0, ".")
    from scripts.build_v5_dry_run import build_v5_dry_run
    build_v5_dry_run()
    print("Rebuilt notebooks/kaggle_dry_run.py from updated shared code!")

if __name__ == "__main__":
    apply_fixes()
