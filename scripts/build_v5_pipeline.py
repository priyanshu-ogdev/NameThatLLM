import re
import sys
from pathlib import Path

def build_v5_ensemble():
    path_ens = Path("notebooks/kaggle_tri_ensemble.py")
    content = path_ens.read_text(encoding="utf-8")

    # 1. Update module docstring & add SCRIPT_T0
    content = content.replace(
        'Name That LLM: Quad-Model Frontier Ensemble (v4 - Hardened Production Pipeline)',
        'Name That LLM: Quad-Model Frontier Ensemble (v5 - Production Hardened Pipeline & Smoke Test)'
    )
    content = content.replace(
        'RichStyleFeatures (132 features):',
        'RichStyleFeatures (134 features):'
    )
    if "SCRIPT_T0 = time.perf_counter()" not in content:
        content = content.replace(
            "import time\n",
            "import time\n\nSCRIPT_T0 = time.perf_counter()\n"
        )

    # 2. Add check_finite_logits and remove nan_to_num from eval functions
    old_eval_block = """    res = sorted_logits[inv_idx]
    return np.nan_to_num(res, nan=0.0)


def evaluate_qwen_split"""
    new_eval_block = """    res = sorted_logits[inv_idx]
    return res


def check_finite_logits(logits: np.ndarray, name: str) -> np.ndarray:
    \"\"\"Validates that a logits matrix contains strictly finite values.\"\"\"
    if logits is None:
        return None
    if not np.all(np.isfinite(logits)):
        n_nan = int(np.sum(np.isnan(logits)))
        n_inf = int(np.sum(np.isinf(logits)))
        raise FloatingPointError(f"[{name}] Non-finite logits detected: {n_nan} NaNs, {n_inf} Infs in shape {logits.shape}")
    return logits


def evaluate_qwen_split"""
    content = content.replace(old_eval_block, new_eval_block)

    old_qwen_eval_ret = """    res = sorted_logits[inv_idx]
    return np.nan_to_num(res, nan=0.0)


def train_transformer_fold"""
    new_qwen_eval_ret = """    res = sorted_logits[inv_idx]
    return res


def train_transformer_fold"""
    content = content.replace(old_qwen_eval_ret, new_qwen_eval_ret)

    # 3. LLRD: word embeddings receive higher LR so newly added [NL], [PAR] can adapt
    old_llrd_emb = """    # Embeddings layer (and relative position embeddings for DeBERTa)
    n_layers = len(layers) if layers is not None else 24
    emb_lr = lr_backbone * (layer_decay ** n_layers)
    emb_params = []"""

    new_llrd_emb = """    # Embeddings layer: word embeddings receive higher LR so newly added [NL], [PAR] tokens can adapt
    n_layers = len(layers) if layers is not None else 24
    emb_lr = max(lr_backbone * (layer_decay ** 4), 8e-6)
    emb_params = []"""
    content = content.replace(old_llrd_emb, new_llrd_emb)

    # 4. Token initialization for [NL] and [PAR] from existing punctuation
    old_token_resize = """    # Add dedicated visible tokens for newlines and resize embeddings
    num_added = tokenizer.add_tokens(["[NL]", "[PAR]"], special_tokens=True)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))"""

    new_token_resize = """    # Add dedicated visible tokens for newlines, resize embeddings, and initialize from punctuation
    num_added = tokenizer.add_tokens(["[NL]", "[PAR]"], special_tokens=True)
    if num_added > 0:
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
    content = content.replace(old_token_resize, new_token_resize)

    # 5. AdamW foreach=False to prevent peak transient memory
    content = content.replace("optimizer = AdamW(opt_params)", "optimizer = AdamW(opt_params, foreach=False)")

    # 6. NaN skip counter & OOM halving in train_transformer_fold
    old_micro_loop = """            try:
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits = model(**batch_kwargs).logits
                    loss = criterion(logits.float(), batch_labels) / grad_accum

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [{model_name}] Warning: NaN loss in micro-step. Skipping micro-batch.", flush=True)
                    if accumulated_steps > 0:
                        optimizer.zero_grad()
                        accumulated_steps = 0
                    continue

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            except torch.cuda.OutOfMemoryError:
                print(f"  [{model_name}] CUDA OOM detected. Enabling gradient checkpointing and retrying micro-batch...", flush=True)
                try:
                    del logits, loss
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    model.gradient_checkpointing_enable()
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits = model(**batch_kwargs).logits
                    loss = criterion(logits.float(), batch_labels) / grad_accum
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()"""

    new_micro_loop = """            try:
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits = model(**batch_kwargs).logits
                    loss = criterion(logits.float(), batch_labels) / grad_accum

                if torch.isnan(loss) or torch.isinf(loss):
                    n_skipped_micros += 1
                    print(f"  [{model_name}] Warning: NaN/Inf loss in micro-step {micro_step}. Skipping micro-batch ({n_skipped_micros} total skips).", flush=True)
                    if n_skipped_micros > 0.15 * max(n_total_micros * (epoch + 1), 1):
                        raise FloatingPointError(f"[{model_name}] >15% micro-batches ({n_skipped_micros}) produced NaN/Inf loss. Unstable FP16 training detected; discarding fold.")
                    if accumulated_steps > 0:
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_steps = 0
                    continue

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            except torch.cuda.OutOfMemoryError:
                print(f"  [{model_name}] CUDA OOM detected. Resetting partial grads, enabling gradient checkpointing, and halving micro-batch...", flush=True)
                try:
                    del logits, loss
                except Exception:
                    pass
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    model.gradient_checkpointing_enable()
                # Split micro-batch into two half-size chunks to guarantee recovery
                half = len(idx) // 2
                sub_slices = [idx[:half], idx[half:]] if half > 0 else [idx]
                for sub_idx in sub_slices:
                    try:
                        sub_sl = {
                            "input_ids": [enc_tr["input_ids"][j] for j in sub_idx],
                            "attention_mask": [enc_tr["attention_mask"][j] for j in sub_idx],
                        }
                        if "token_type_ids" in enc_tr:
                            sub_sl["token_type_ids"] = [enc_tr["token_type_ids"][j] for j in sub_idx]
                        sub_padded = tokenizer.pad(sub_sl, padding=True, return_tensors="pt")
                        sub_kw = {
                            "input_ids": sub_padded["input_ids"].to(dev, non_blocking=True),
                            "attention_mask": sub_padded["attention_mask"].to(dev, non_blocking=True),
                        }
                        if "token_type_ids" in sub_padded:
                            sub_kw["token_type_ids"] = sub_padded["token_type_ids"].to(dev, non_blocking=True)
                        sub_labels = torch.tensor(y_tr_int[sub_idx], dtype=torch.long, device=dev)
                        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                            sub_logits = model(**sub_kw).logits
                            sub_loss = criterion(sub_logits.float(), sub_labels) / (grad_accum * len(sub_slices))
                        if use_amp:
                            scaler.scale(sub_loss).backward()
                        else:
                            sub_loss.backward()
                    except torch.cuda.OutOfMemoryError:
                        print(f"  [{model_name}] Second OOM in sub-batch: skipping sample to preserve fold.", flush=True)
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_steps = 0
                        continue"""
    content = content.replace(old_micro_loop, new_micro_loop)

    content = content.replace(
        "stop_early = False\n\n    for epoch in range(epochs):",
        "stop_early = False\n    n_skipped_micros = 0\n\n    for epoch in range(epochs):",
        1
    )

    content = content.replace(
        "val_logits = evaluate_transformer_split(model, enc_va, tokenizer, dev, use_amp, len(labels))",
        "val_logits = evaluate_transformer_split(model, enc_va, tokenizer, dev, use_amp, len(labels))\n        check_finite_logits(val_logits, f'{model_name} Val')"
    )
    content = content.replace(
        "best_test_logits = evaluate_transformer_split(model, enc_te, tokenizer, dev, use_amp, len(labels))",
        "best_test_logits = evaluate_transformer_split(model, enc_te, tokenizer, dev, use_amp, len(labels))\n    check_finite_logits(best_test_logits, f'{model_name} Test')"
    )

    # In train_qwen_lora_fold:
    content = content.replace(
        "stop_early = False\n\n    for epoch in range(epochs):",
        "stop_early = False\n    n_skipped_micros = 0\n\n    for epoch in range(epochs):",
        1
    )
    content = content.replace(
        "val_logits = evaluate_qwen_split(backbone, head, enc_va, tokenizer, first_dev, head_dev, cuda_ok, len(labels))",
        "val_logits = evaluate_qwen_split(backbone, head, enc_va, tokenizer, first_dev, head_dev, cuda_ok, len(labels))\n        check_finite_logits(val_logits, 'Qwen Val')"
    )
    content = content.replace(
        "best_test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels))",
        "best_test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels))\n    check_finite_logits(best_test_logits, 'Qwen Test')"
    )

    # 7. Add fit_temperature function to ensemble utilities
    fit_temp_func = """def fit_temperature(margins: np.ndarray, y_true_int: np.ndarray, default_t: float = 1.0) -> float:
    \"\"\"
    Fits optimal Platt/temperature scaling parameter T by minimizing NLL on valid OOF margins.
    Searches over [0.05, 3.0] using numerical log-sum-exp; guaranteed non-zero and stable.
    \"\"\"
    if margins is None:
        return default_t
    valid = np.isfinite(margins).all(axis=1) & (y_true_int >= 0)
    if np.sum(valid) < 20:
        return default_t
    m_val = margins[valid]
    y_val = y_true_int[valid]

    best_t = default_t
    best_nll = float("inf")
    for t_cand in np.linspace(0.05, 3.0, 60):
        scaled = m_val / t_cand
        max_s = np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        sum_exp = np.sum(exp_s, axis=1, keepdims=True)
        log_probs = (scaled - max_s) - np.log(sum_exp + 1e-12)
        nll = -np.mean(log_probs[np.arange(len(y_val)), y_val])
        if nll < best_nll:
            best_nll = nll
            best_t = float(t_cand)
    return round(best_t, 3)


def margins_to_probs"""
    if "def fit_temperature(" not in content:
        content = content.replace("def margins_to_probs", fit_temp_func)

    # 8. Update quad_rank_blend and optimize_blend_weights to accept temperatures
    old_quad_sig = """def quad_rank_blend(
    svm_m: np.ndarray,
    deberta_m: np.ndarray = None,
    qwen_m: np.ndarray = None,
    bge_m: np.ndarray = None,
    weights=(1.0, 1.0, 1.0, 1.0),
    balance_test: bool = False,
    apply_family_gating: bool = True,
):"""
    new_quad_sig = """def quad_rank_blend(
    svm_m: np.ndarray,
    deberta_m: np.ndarray = None,
    qwen_m: np.ndarray = None,
    bge_m: np.ndarray = None,
    weights=(1.0, 1.0, 1.0, 1.0),
    temperatures=(0.20, 1.2, 1.5, 1.2),
    balance_test: bool = False,
    apply_family_gating: bool = True,
):"""
    content = content.replace(old_quad_sig, new_quad_sig)

    content = content.replace(
        "total_prob = ws * margins_to_probs(svm_m, temp=0.20)",
        "ts, td, tq, tb = temperatures\n    total_prob = ws * margins_to_probs(svm_m, temp=ts)"
    )
    content = content.replace(
        "total_prob = total_prob + wd * margins_to_probs(deberta_m, temp=1.2)",
        "total_prob = total_prob + wd * margins_to_probs(deberta_m, temp=td)"
    )
    content = content.replace(
        "total_prob = total_prob + wq * margins_to_probs(qwen_m, temp=1.5)",
        "total_prob = total_prob + wq * margins_to_probs(qwen_m, temp=tq)"
    )
    content = content.replace(
        "total_prob = total_prob + wb * margins_to_probs(bge_m, temp=1.2)",
        "total_prob = total_prob + wb * margins_to_probs(bge_m, temp=tb)"
    )

    old_opt_sig = """def optimize_blend_weights(oof_svm, oof_deb, oof_qwen, oof_bge, y_true):"""
    new_opt_sig = """def optimize_blend_weights(oof_svm, oof_deb, oof_qwen, oof_bge, y_true, temperatures=(0.20, 1.2, 1.5, 1.2)):"""
    content = content.replace(old_opt_sig, new_opt_sig)

    content = content.replace(
        """    svm_p = margins_to_probs(oof_svm[valid_mask], temp=0.20)
    deb_p = margins_to_probs(oof_deb[valid_mask], temp=1.2) if oof_deb is not None else None
    qwn_p = margins_to_probs(oof_qwen[valid_mask], temp=1.5) if oof_qwen is not None else None
    bge_p = margins_to_probs(oof_bge[valid_mask], temp=1.2) if oof_bge is not None else None""",
        """    ts, td, tq, tb = temperatures
    svm_p = margins_to_probs(oof_svm[valid_mask], temp=ts)
    deb_p = margins_to_probs(oof_deb[valid_mask], temp=td) if oof_deb is not None else None
    qwn_p = margins_to_probs(oof_qwen[valid_mask], temp=tq) if oof_qwen is not None else None
    bge_p = margins_to_probs(oof_bge[valid_mask], temp=tb) if oof_bge is not None else None"""
    )

    # 9. make_prompt_groups: str(text) defense
    content = content.replace(
        'h = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()',
        'h = hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()'
    )

    # 10. Update run_quad_ensemble in kaggle_tri_ensemble.py
    # Banner
    content = content.replace(
        'print("NAME THAT LLM: QUAD-MODEL FRONTIER ENSEMBLE (v3 - Max Hardware)", flush=True)',
        'print("NAME THAT LLM: QUAD-MODEL FRONTIER ENSEMBLE (v5 - Production Pipeline)", flush=True)'
    )
    # Fillna immediately after reading CSV
    content = content.replace(
        '    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str, "prompt_id": str})\n    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})\n',
        '    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str, "prompt_id": str})\n    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})\n    train_df["text"] = train_df["text"].fillna("")\n    test_df["text"] = test_df["text"].fillna("")\n'
    )
    # Clock measured from SCRIPT_T0
    content = content.replace(
        """    pipeline_t0 = time.perf_counter()
    pipeline_deadline = pipeline_t0 + (MAX_WALLCLOCK_HOURS * 3600) - SAFETY_MARGIN_SEC""",
        """    pipeline_t0 = SCRIPT_T0
    est_inference_sec = 900.0  # 15 min reserve for final inference, calibration and blending
    pipeline_deadline = SCRIPT_T0 + (MAX_WALLCLOCK_HOURS * 3600) - SAFETY_MARGIN_SEC - est_inference_sec"""
    )

    # Wrap CPU fold in try/except
    old_cpu_fold = """        # ── 4. CPU Multiview Dual-Linear (Fast pre-extracted sparse slice) ──
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

    new_cpu_fold = """        # ── 4. CPU Multiview Dual-Linear (Fast pre-extracted sparse slice) ──
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
                print(f"  [CPU Linear] Fold {fold+1} bal_acc={svm_bal_acc*100:.2f}%", flush=True)
        except Exception as e:
            import traceback
            print(f"  [CPU Linear] Fold {fold+1} error: {e}", flush=True)
            traceback.print_exc()"""
    content = content.replace(old_cpu_fold, new_cpu_fold)

    # Wrap interim submission checkpoint in try/except
    old_interim_sub = """        # ── Safe Incremental Submission Checkpoint ────────────
        # Guarantees a fully valid, competition-compliant submission.parquet exists on disk after EVERY fold
        n_test = len(texts_test)
        interim_svm_m = np.mean(test_svm_folds, axis=0) if test_svm_folds else np.zeros((n_test, NUM_CLASSES))
        interim_deb_m = np.mean(test_deb_folds, axis=0) if test_deb_folds else None
        interim_bge_m = np.mean(test_bge_folds, axis=0) if test_bge_folds else None
        interim_qwen_m = np.mean(test_qwen_folds, axis=0) if test_qwen_folds else None

        interim_preds_idx = quad_rank_blend(
            interim_svm_m, interim_deb_m, interim_qwen_m, interim_bge_m,
            weights=(0.20, 0.35, 0.20, 0.25),
            balance_test=True,
            apply_family_gating=True,
        )
        validate_and_save_submission(
            test_df["id"].tolist(),
            [LABELS[i] for i in interim_preds_idx],
            working_dir / "submission.parquet",
        )
        print(f"  [Safety Checkpoint] Incremental submission.parquet saved after Fold {fold+1}/{n_splits} [OK]", flush=True)"""

    new_interim_sub = """        # ── Safe Incremental Submission Checkpoint ────────────
        # Guarantees a fully valid, competition-compliant submission.parquet exists on disk after EVERY fold
        try:
            n_test = len(texts_test)
            interim_svm_m = np.mean(test_svm_folds, axis=0) if test_svm_folds else np.zeros((n_test, NUM_CLASSES))
            interim_deb_m = np.mean(test_deb_folds, axis=0) if test_deb_folds else None
            interim_bge_m = np.mean(test_bge_folds, axis=0) if test_bge_folds else None
            interim_qwen_m = np.mean(test_qwen_folds, axis=0) if test_qwen_folds else None

            interim_preds_idx = quad_rank_blend(
                interim_svm_m, interim_deb_m, interim_qwen_m, interim_bge_m,
                weights=(0.20, 0.35, 0.20, 0.25),
                temperatures=(0.20, 1.2, 1.5, 1.2),
                balance_test=True,
                apply_family_gating=True,
            )
            validate_and_save_submission(
                test_df["id"].tolist(),
                [LABELS[i] for i in interim_preds_idx],
                working_dir / "submission.parquet",
            )
            print(f"  [Safety Checkpoint] Incremental submission.parquet saved after Fold {fold+1}/{n_splits} [OK]", flush=True)
        except Exception as e:
            import traceback
            print(f"  [Safety Checkpoint] Warning during interim submission write: {e}", flush=True)
            traceback.print_exc()"""
    content = content.replace(old_interim_sub, new_interim_sub)

    # Universal Neural Stream Quality Gate (DeBERTa, BGE, Qwen) + Temperature Calibration + Diagnostic Stacker mask
    old_oof_eval = """    eval_mask = ~np.isnan(oof_svm[:, 0])
    y_eval = y_true[eval_mask]

    svm_score = balanced_accuracy_score(y_eval, [LABELS[i] for i in np.argmax(oof_svm[eval_mask], axis=1)])
    print(f"  CPU Dual-Linear: {svm_score*100:.2f}%", flush=True)

    if oof_deb is not None and np.any(~np.isnan(oof_deb[:, 0])):
        deb_mask = ~np.isnan(oof_deb[:, 0])
        deb_score = balanced_accuracy_score(y_true[deb_mask], [LABELS[i] for i in np.argmax(oof_deb[deb_mask], axis=1)])
        print(f"  DeBERTa FT:      {deb_score*100:.2f}% (folds evaluated: {np.sum(deb_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_deb = None
        print("  DeBERTa FT:      (skipped)", flush=True)

    if oof_bge is not None and np.any(~np.isnan(oof_bge[:, 0])):
        bge_mask = ~np.isnan(oof_bge[:, 0])
        bge_score = balanced_accuracy_score(y_true[bge_mask], [LABELS[i] for i in np.argmax(oof_bge[bge_mask], axis=1)])
        print(f"  BGE FT:          {bge_score*100:.2f}% (folds evaluated: {np.sum(bge_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_bge = None
        print("  BGE FT:          (skipped)", flush=True)

    if oof_qwen is not None and np.any(~np.isnan(oof_qwen).any(axis=1)):
        qwen_mask = ~np.isnan(oof_qwen).any(axis=1)
        qwen_score = balanced_accuracy_score(y_true[qwen_mask], [LABELS[i] for i in np.argmax(oof_qwen[qwen_mask], axis=1)])
        if qwen_score < 0.08:
            print(f"  [Qwen LoRA] Warning: OOF score ({qwen_score*100:.2f}%) near random chance. Dropping stream from blend.", flush=True)
            oof_qwen = None
            test_qwen_folds = []
        else:
            print(f"  Qwen LoRA:       {qwen_score*100:.2f}% (folds evaluated: {np.sum(qwen_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_qwen = None
        print("  Qwen LoRA:       (skipped)", flush=True)

    best_w, blend_score = optimize_blend_weights(
        oof_svm,
        oof_deb,
        oof_qwen,
        oof_bge,
        y_true,
    )
    ws, wd, wq, wb = best_w
    print(f"\\n  Blend weights: CPU={ws} DeBERTa={wd} Qwen={wq} BGE={wb}", flush=True)
    print(f"  OOF QUAD BLEND: {blend_score*100:.2f}%\\n", flush=True)

    # Diagnostic: Ridge Stacker Comparison (5-Fold CV out-of-sample)
    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import cross_val_predict
        avail_streams = [margins_to_probs(oof_svm, temp=0.20)]
        if oof_deb is not None: avail_streams.append(margins_to_probs(oof_deb, temp=1.2))
        if oof_qwen is not None: avail_streams.append(margins_to_probs(oof_qwen, temp=1.5))
        if oof_bge is not None: avail_streams.append(margins_to_probs(oof_bge, temp=1.2))
        if len(avail_streams) > 1:
            stack_mask = ~np.isnan(avail_streams[0]).any(axis=1)
            for st in avail_streams[1:]:
                stack_mask = stack_mask & ~np.isnan(st).any(axis=1)
            if np.any(stack_mask):
                lbl2id = {l: i for i, l in enumerate(LABELS)}
                y_stack = np.array([lbl2id[y] for y in y_true[stack_mask]])
                X_oof_stack = np.hstack([st[stack_mask] for st in avail_streams])
                ridge_preds = cross_val_predict(RidgeClassifier(alpha=1.0), X_oof_stack, y_stack, cv=5)
                ridge_score = balanced_accuracy_score(y_stack, ridge_preds)
                print(f"  OOF RIDGE STACKER (5-Fold CV): {ridge_score*100:.2f}% (out-of-sample diagnostic)\\n", flush=True)
    except Exception as e:
        print(f"  [Diagnostic Stacker] Note: {e}", flush=True)

    # --- 5-fold bagged test predictions ---
    print("COMPUTING BAGGED TEST PREDICTIONS", flush=True)
    n_test = len(texts_test)
    test_svm_m = np.mean(test_svm_folds, axis=0) if test_svm_folds else np.zeros((n_test, NUM_CLASSES))
    test_deb_m = np.mean(test_deb_folds, axis=0) if test_deb_folds else None
    test_bge_m = np.mean(test_bge_folds, axis=0) if test_bge_folds else None
    test_qwen_m = np.mean(test_qwen_folds, axis=0) if test_qwen_folds else None

    test_pred_idx = quad_rank_blend(
        test_svm_m, test_deb_m, test_qwen_m, test_bge_m,
        weights=best_w,
        balance_test=True,
        apply_family_gating=True,
    )
    test_predictions = [LABELS[i] for i in test_pred_idx]

    sub_path = working_dir / "submission.parquet"
    validate_and_save_submission(test_df["id"].tolist(), test_predictions, sub_path)"""

    new_oof_eval = """    eval_mask = ~np.isnan(oof_svm[:, 0])
    y_eval = y_true[eval_mask]

    svm_score = balanced_accuracy_score(y_eval, [LABELS[i] for i in np.argmax(oof_svm[eval_mask], axis=1)])
    print(f"  CPU Dual-Linear: {svm_score*100:.2f}%", flush=True)

    # Universal Neural Stream Quality Gate (relative to CPU baseline)
    min_acceptable_score = max(1.5 / NUM_CLASSES, 0.5 * svm_score)

    if oof_deb is not None and np.any(~np.isnan(oof_deb[:, 0])):
        deb_mask = ~np.isnan(oof_deb[:, 0])
        deb_score = balanced_accuracy_score(y_true[deb_mask], [LABELS[i] for i in np.argmax(oof_deb[deb_mask], axis=1)])
        if deb_score < min_acceptable_score:
            print(f"  [DeBERTa FT] Warning: OOF score ({deb_score*100:.2f}%) below minimum quality threshold ({min_acceptable_score*100:.2f}%). Dropping stream from blend.", flush=True)
            oof_deb = None
            test_deb_folds = []
        else:
            print(f"  DeBERTa FT:      {deb_score*100:.2f}% (folds evaluated: {np.sum(deb_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_deb = None
        print("  DeBERTa FT:      (skipped)", flush=True)

    if oof_bge is not None and np.any(~np.isnan(oof_bge[:, 0])):
        bge_mask = ~np.isnan(oof_bge[:, 0])
        bge_score = balanced_accuracy_score(y_true[bge_mask], [LABELS[i] for i in np.argmax(oof_bge[bge_mask], axis=1)])
        if bge_score < min_acceptable_score:
            print(f"  [BGE FT] Warning: OOF score ({bge_score*100:.2f}%) below minimum quality threshold ({min_acceptable_score*100:.2f}%). Dropping stream from blend.", flush=True)
            oof_bge = None
            test_bge_folds = []
        else:
            print(f"  BGE FT:          {bge_score*100:.2f}% (folds evaluated: {np.sum(bge_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_bge = None
        print("  BGE FT:          (skipped)", flush=True)

    if oof_qwen is not None and np.any(~np.isnan(oof_qwen).any(axis=1)):
        qwen_mask = ~np.isnan(oof_qwen).any(axis=1)
        qwen_score = balanced_accuracy_score(y_true[qwen_mask], [LABELS[i] for i in np.argmax(oof_qwen[qwen_mask], axis=1)])
        if qwen_score < min_acceptable_score:
            print(f"  [Qwen LoRA] Warning: OOF score ({qwen_score*100:.2f}%) below minimum quality threshold ({min_acceptable_score*100:.2f}%). Dropping stream from blend.", flush=True)
            oof_qwen = None
            test_qwen_folds = []
        else:
            print(f"  Qwen LoRA:       {qwen_score*100:.2f}% (folds evaluated: {np.sum(qwen_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_qwen = None
        print("  Qwen LoRA:       (skipped)", flush=True)

    # Automated Temperature Calibration (minimizing NLL on valid OOF rows)
    lbl2id = {l: i for i, l in enumerate(LABELS)}
    y_true_int = np.array([lbl2id[y] for y in y_true], dtype=np.int64)
    T_svm = fit_temperature(oof_svm, y_true_int, default_t=0.20)
    T_deb = fit_temperature(oof_deb, y_true_int, default_t=1.20) if oof_deb is not None else 1.20
    T_qwn = fit_temperature(oof_qwen, y_true_int, default_t=1.50) if oof_qwen is not None else 1.50
    T_bge = fit_temperature(oof_bge, y_true_int, default_t=1.20) if oof_bge is not None else 1.20
    calibrated_temps = (T_svm, T_deb, T_qwn, T_bge)
    print(f"\\n  Calibrated Temperatures (NLL-optimal): SVM={T_svm} DeBERTa={T_deb} Qwen={T_qwn} BGE={T_bge}", flush=True)

    try:
        best_w, blend_score = optimize_blend_weights(
            oof_svm,
            oof_deb,
            oof_qwen,
            oof_bge,
            y_true,
            temperatures=calibrated_temps,
        )
        ws, wd, wq, wb = best_w
        print(f"\\n  Blend weights: CPU={ws} DeBERTa={wd} Qwen={wq} BGE={wb}", flush=True)
        print(f"  OOF QUAD BLEND: {blend_score*100:.2f}%\\n", flush=True)
    except Exception as e:
        import traceback
        print(f"  [Blend Tuning] Error during blend optimization: {e}", flush=True)
        traceback.print_exc()
        best_w = (1.0, 1.0, 1.0, 1.0)

    # Diagnostic: Ridge Stacker Comparison (5-Fold CV out-of-sample)
    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import cross_val_predict
        # Compute mask directly from raw OOF arrays before margins_to_probs replaces NaNs
        raw_streams = [oof_svm]
        if oof_deb is not None: raw_streams.append(oof_deb)
        if oof_qwen is not None: raw_streams.append(oof_qwen)
        if oof_bge is not None: raw_streams.append(oof_bge)
        if len(raw_streams) > 1:
            stack_mask = ~np.isnan(raw_streams[0]).any(axis=1)
            for raw_st in raw_streams[1:]:
                stack_mask = stack_mask & ~np.isnan(raw_st).any(axis=1)
            if np.any(stack_mask):
                y_stack = y_true_int[stack_mask]
                avail_probs = [margins_to_probs(oof_svm[stack_mask], temp=T_svm)]
                if oof_deb is not None: avail_probs.append(margins_to_probs(oof_deb[stack_mask], temp=T_deb))
                if oof_qwen is not None: avail_probs.append(margins_to_probs(oof_qwen[stack_mask], temp=T_qwn))
                if oof_bge is not None: avail_probs.append(margins_to_probs(oof_bge[stack_mask], temp=T_bge))
                X_oof_stack = np.hstack(avail_probs)
                ridge_preds = cross_val_predict(RidgeClassifier(alpha=1.0), X_oof_stack, y_stack, cv=5)
                ridge_score = balanced_accuracy_score(y_stack, ridge_preds)
                print(f"  OOF RIDGE STACKER (5-Fold CV): {ridge_score*100:.2f}% (out-of-sample diagnostic)\\n", flush=True)
    except Exception as e:
        print(f"  [Diagnostic Stacker] Note: {e}", flush=True)

    # --- 5-fold bagged test predictions ---
    try:
        print("COMPUTING BAGGED TEST PREDICTIONS", flush=True)
        n_test = len(texts_test)
        test_svm_m = np.mean(test_svm_folds, axis=0) if test_svm_folds else np.zeros((n_test, NUM_CLASSES))
        test_deb_m = np.mean(test_deb_folds, axis=0) if test_deb_folds else None
        test_bge_m = np.mean(test_bge_folds, axis=0) if test_bge_folds else None
        test_qwen_m = np.mean(test_qwen_folds, axis=0) if test_qwen_folds else None

        test_pred_idx = quad_rank_blend(
            test_svm_m, test_deb_m, test_qwen_m, test_bge_m,
            weights=best_w,
            temperatures=calibrated_temps,
            balance_test=True,
            apply_family_gating=True,
        )
        test_predictions = [LABELS[i] for i in test_pred_idx]

        sub_path = working_dir / "submission.parquet"
        validate_and_save_submission(test_df["id"].tolist(), test_predictions, sub_path)
    except Exception as e:
        import traceback
        print(f"  [Final Blend] Error during final test prediction / submission write: {e}", flush=True)
        traceback.print_exc()
        print("  [Final Blend] Preserving last successful incremental submission.parquet on disk.", flush=True)"""
    content = content.replace(old_oof_eval, new_oof_eval)

    path_ens.write_text(content, encoding="utf-8")
    print(f"Updated {path_ens} successfully.")
    return content

build_v5_ensemble()

