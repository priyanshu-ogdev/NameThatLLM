import re
from pathlib import Path

path = Path("notebooks/kaggle_tri_ensemble.py")
content = path.read_text(encoding="utf-8")

# ==============================================================================
# 1. Update train_qwen_lora_fold:
#    - raw_text (preserve_newlines=False)
#    - length-sorted infer_val and infer_test with inverse permutation
#    - intra-epoch deadline check inside micro-batch loop
#    - check scaler before scheduler step
#    - n_gpus_local in cleanup
# ==============================================================================
target_qwen_tokenization = """    print(f"  [Qwen LoRA] Tokenizing {len(texts_tr)} train / {len(texts_va)} val / {len(texts_te)} test passages (Head+Tail L={max_length}, [NL] structure preserved)...", flush=True)
    enc_tr = tokenize_with_head_tail(tokenizer, texts_tr, max_length=max_length)
    enc_va = tokenize_with_head_tail(tokenizer, texts_va, max_length=max_length)
    enc_te = tokenize_with_head_tail(tokenizer, texts_te, max_length=max_length)"""

repl_qwen_tokenization = """    print(f"  [Qwen LoRA] Tokenizing {len(texts_tr)} train / {len(texts_va)} val / {len(texts_te)} test passages (Head+Tail L={max_length}, native BPE newlines preserved)...", flush=True)
    enc_tr = tokenize_with_head_tail(tokenizer, texts_tr, max_length=max_length, preserve_newlines=False)
    enc_va = tokenize_with_head_tail(tokenizer, texts_va, max_length=max_length, preserve_newlines=False)
    enc_te = tokenize_with_head_tail(tokenizer, texts_te, max_length=max_length, preserve_newlines=False)"""

assert target_qwen_tokenization in content, "target_qwen_tokenization not found"
content = content.replace(target_qwen_tokenization, repl_qwen_tokenization, 1)

target_qwen_infer = """    def infer_val():
        if len(texts_va) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        backbone.eval()
        head.eval()
        all_logits = []
        inf_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_va), inf_bs):
                end = min(start + inf_bs, len(texts_va))
                sl = {
                    "input_ids": enc_va["input_ids"][start:end],
                    "attention_mask": enc_va["attention_mask"][start:end],
                }
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                batch_kwargs = {
                    "input_ids": padded["input_ids"].to(first_dev, non_blocking=True),
                    "attention_mask": padded["attention_mask"].to(first_dev, non_blocking=True),
                }
                if cuda_ok:
                    with amp_autocast():
                        out = backbone(**batch_kwargs, output_hidden_states=False)
                else:
                    out = backbone(**batch_kwargs, output_hidden_states=False)

                mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
                hidden_f = out.last_hidden_state.float()
                pooled = (hidden_f * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                logits = head(pooled.to(head_dev)).float().cpu().numpy()
                all_logits.append(logits)
        return np.concatenate(all_logits, axis=0)

    def infer_test():
        if len(texts_te) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        backbone.eval()
        head.eval()
        all_logits = []
        inf_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_te), inf_bs):
                end = min(start + inf_bs, len(texts_te))
                sl = {
                    "input_ids": enc_te["input_ids"][start:end],
                    "attention_mask": enc_te["attention_mask"][start:end],
                }
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                batch_kwargs = {
                    "input_ids": padded["input_ids"].to(first_dev, non_blocking=True),
                    "attention_mask": padded["attention_mask"].to(first_dev, non_blocking=True),
                }
                if cuda_ok:
                    with amp_autocast():
                        out = backbone(**batch_kwargs, output_hidden_states=False)
                else:
                    out = backbone(**batch_kwargs, output_hidden_states=False)

                mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
                hidden_f = out.last_hidden_state.float()
                pooled = (hidden_f * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                logits = head(pooled.to(head_dev)).float().cpu().numpy()
                all_logits.append(logits)
        return np.concatenate(all_logits, axis=0)"""

repl_qwen_infer = """    def infer_val():
        if len(texts_va) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        backbone.eval()
        head.eval()
        va_lengths = [len(x) for x in enc_va["input_ids"]]
        sort_idx = np.argsort(va_lengths)
        all_logits = []
        inf_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_va), inf_bs):
                end = min(start + inf_bs, len(texts_va))
                batch_indices = sort_idx[start:end]
                sl = {
                    "input_ids": [enc_va["input_ids"][j] for j in batch_indices],
                    "attention_mask": [enc_va["attention_mask"][j] for j in batch_indices],
                }
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                batch_kwargs = {
                    "input_ids": padded["input_ids"].to(first_dev, non_blocking=True),
                    "attention_mask": padded["attention_mask"].to(first_dev, non_blocking=True),
                }
                if cuda_ok:
                    with amp_autocast():
                        out = backbone(**batch_kwargs, output_hidden_states=False)
                else:
                    out = backbone(**batch_kwargs, output_hidden_states=False)

                mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
                hidden_f = out.last_hidden_state.float()
                pooled = (hidden_f * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                logits = head(pooled.to(head_dev)).float().cpu().numpy()
                all_logits.append(logits)
        sorted_logits = np.concatenate(all_logits, axis=0)
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(len(sort_idx))
        res = sorted_logits[inv_idx]
        return np.nan_to_num(res, nan=0.0)

    def infer_test():
        if len(texts_te) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        backbone.eval()
        head.eval()
        te_lengths = [len(x) for x in enc_te["input_ids"]]
        sort_idx = np.argsort(te_lengths)
        all_logits = []
        inf_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_te), inf_bs):
                end = min(start + inf_bs, len(texts_te))
                batch_indices = sort_idx[start:end]
                sl = {
                    "input_ids": [enc_te["input_ids"][j] for j in batch_indices],
                    "attention_mask": [enc_te["attention_mask"][j] for j in batch_indices],
                }
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                batch_kwargs = {
                    "input_ids": padded["input_ids"].to(first_dev, non_blocking=True),
                    "attention_mask": padded["attention_mask"].to(first_dev, non_blocking=True),
                }
                if cuda_ok:
                    with amp_autocast():
                        out = backbone(**batch_kwargs, output_hidden_states=False)
                else:
                    out = backbone(**batch_kwargs, output_hidden_states=False)

                mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
                hidden_f = out.last_hidden_state.float()
                pooled = (hidden_f * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                logits = head(pooled.to(head_dev)).float().cpu().numpy()
                all_logits.append(logits)
        sorted_logits = np.concatenate(all_logits, axis=0)
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(len(sort_idx))
        res = sorted_logits[inv_idx]
        return np.nan_to_num(res, nan=0.0)"""

assert target_qwen_infer in content, "target_qwen_infer not found"
content = content.replace(target_qwen_infer, repl_qwen_infer, 1)

target_qwen_loop = """        while i < len(texts_tr):
            idx = epoch_indices[i:i + curr_batch]"""

repl_qwen_loop = """        stop_early = False
        while i < len(texts_tr):
            if deadline is not None and (n_batches % 10 == 0) and time.perf_counter() > deadline:
                print(f"  [Qwen LoRA] [Time Guard] Deadline reached mid-epoch at step {n_batches}/{n_total_micros}. Stopping training.", flush=True)
                stop_early = True
                break

            idx = epoch_indices[i:i + curr_batch]"""

assert target_qwen_loop in content, "target_qwen_loop not found"
content = content.replace(target_qwen_loop, repl_qwen_loop, 1)

target_qwen_step = """                if n_batches % grad_accum == 0 or last_micro:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()"""

repl_qwen_step = """                if n_batches % grad_accum == 0 or last_micro:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = scaler.get_scale()
                    if new_scale >= old_scale:
                        scheduler.step()
                    optimizer.zero_grad()"""

assert target_qwen_step in content, "target_qwen_step not found"
content = content.replace(target_qwen_step, repl_qwen_step, 1)

target_qwen_stop = """        if deadline is not None and time.perf_counter() > deadline:
            print(f"  [Qwen LoRA] [Time Guard] Deadline reached after epoch {epoch+1}/{epochs}. Stopping early to preserve predictions.", flush=True)
            break"""

repl_qwen_stop = """        if stop_early or (deadline is not None and time.perf_counter() > deadline):
            print(f"  [Qwen LoRA] [Time Guard] Stopping early to preserve predictions.", flush=True)
            break"""

assert target_qwen_stop in content, "target_qwen_stop not found"
content = content.replace(target_qwen_stop, repl_qwen_stop, 1)

# Cleanup in Qwen
target_qwen_clean = """    del backbone, head, tokenizer, optimizer, scheduler, weight_tensor, enc_tr, enc_va, enc_te, best_lora_state, best_head_state
    gc.collect()
    if cuda_ok:
        for gi in range(n_gpus):
            with torch.cuda.device(gi):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()"""

repl_qwen_clean = """    n_gpus_local = torch.cuda.device_count() if cuda_ok else 0
    del backbone, head, tokenizer, optimizer, scheduler, weight_tensor, enc_tr, enc_va, enc_te, best_lora_state, best_head_state
    gc.collect()
    if cuda_ok:
        for gi in range(n_gpus_local):
            with torch.cuda.device(gi):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()"""

assert target_qwen_clean in content, "target_qwen_clean not found"
content = content.replace(target_qwen_clean, repl_qwen_clean, 1)

# ==============================================================================
# 2. Update Ensemble Utilities:
#    - remove dead compute_passage_ranks and normalize_margins
#    - NaN-proof margins_to_probs with fallback to uniform 1/K
#    - non-negative and finite assert in Sinkhorn
#    - optimize_blend_weights with any(axis=1) and Sinkhorn active
# ==============================================================================
target_utils = """def compute_passage_ranks(margins: np.ndarray) -> np.ndarray:
    \"\"\"Compute ordinal ranks per row [1, K]. Vectorized via scipy.stats.rankdata.\"\"\"
    return rankdata(margins, axis=1)


def normalize_margins(margins: np.ndarray) -> np.ndarray:
    \"\"\"Zero-mean, unit-variance normalization per row.\"\"\"
    m_mean = margins.mean(axis=1, keepdims=True)
    m_std = np.maximum(margins.std(axis=1, keepdims=True), 1e-8)
    return (margins - m_mean) / m_std


def margins_to_probs(margins: np.ndarray, temp: float = 1.0) -> np.ndarray:
    \"\"\"Convert unbounded margins/logits into calibrated probability distribution.\"\"\"
    clean = np.clip(margins, -50.0, 50.0)
    scaled = clean / max(temp, 1e-4)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / np.maximum(np.sum(exp_s, axis=1, keepdims=True), 1e-12)"""

repl_utils = """def margins_to_probs(margins: np.ndarray, temp: float = 1.0) -> np.ndarray:
    \"\"\"
    Convert unbounded margins/logits into calibrated probability distribution.
    Hardened against NaNs: any row containing non-finite values is replaced with a uniform prior (1/K).
    \"\"\"
    m = np.asarray(margins, dtype=np.float64)
    bad = ~np.isfinite(m).all(axis=1)
    m = np.nan_to_num(m, nan=0.0, posinf=50.0, neginf=-50.0)
    z = np.clip(m, -50.0, 50.0) / max(temp, 1e-4)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    if np.any(bad):
        p[bad] = 1.0 / p.shape[1]
    return p"""

assert target_utils in content, "target_utils not found"
content = content.replace(target_utils, repl_utils, 1)

target_sinkhorn_assert = """    if np.any(probs < 0.0):
        probs = margins_to_probs(probs)"""

repl_sinkhorn_assert = """    if np.any(probs < 0.0) or not np.all(np.isfinite(probs)):
        probs = margins_to_probs(probs)
    assert np.all(np.isfinite(probs)), "Non-finite probabilities reached Sinkhorn balancing\""""

assert target_sinkhorn_assert in content, "target_sinkhorn_assert not found"
content = content.replace(target_sinkhorn_assert, repl_sinkhorn_assert, 1)

target_valid_mask = """    valid_mask = ~np.isnan(oof_svm[:, 0])
    if oof_deb is not None:
        valid_mask &= ~np.isnan(oof_deb[:, 0])
    if oof_bge is not None:
        valid_mask &= ~np.isnan(oof_bge[:, 0])
    if oof_qwen is not None:
        valid_mask &= ~np.isnan(oof_qwen[:, 0])"""

repl_valid_mask = """    valid_mask = ~np.isnan(oof_svm).any(axis=1)
    if oof_deb is not None:
        valid_mask &= ~np.isnan(oof_deb).any(axis=1)
    if oof_bge is not None:
        valid_mask &= ~np.isnan(oof_bge).any(axis=1)
    if oof_qwen is not None:
        valid_mask &= ~np.isnan(oof_qwen).any(axis=1)"""

assert target_valid_mask in content, "target_valid_mask not found"
content = content.replace(target_valid_mask, repl_valid_mask, 1)

target_blend_eval = """                    gated = apply_hierarchical_family_gating(blend_p, alpha=0.35)
                    pred_idx = np.argmax(gated, axis=1)"""

repl_blend_eval = """                    gated = apply_hierarchical_family_gating(blend_p, alpha=0.35)
                    if len(gated) >= 36:
                        gated = sinkhorn_knopp_balance(gated)
                    pred_idx = np.argmax(gated, axis=1)"""

assert target_blend_eval in content, "target_blend_eval not found"
content = content.replace(target_blend_eval, repl_blend_eval, 1)

# ==============================================================================
# 3. Step 0 One-Time fit_transform & Stage Duration Pre-flight Checks
# ==============================================================================
target_step0 = """    feature_extractor = build_cpu_feature_extractor(style_weight=0.2)
    all_raw_texts = raw_texts_tr + raw_texts_te
    with threadpool_limits(limits=4):
        feature_extractor.fit(all_raw_texts)
        X_train_all = feature_extractor.transform(raw_texts_tr)
        X_test_all = feature_extractor.transform(raw_texts_te)"""

repl_step0 = """    print("Train class distribution:")
    print(train_df["model"].value_counts().to_string(), flush=True)

    feature_extractor = build_cpu_feature_extractor(style_weight=0.2)
    all_raw_texts = raw_texts_tr + raw_texts_te
    n_tr = len(raw_texts_tr)
    with threadpool_limits(limits=4):
        X_all = feature_extractor.fit_transform(all_raw_texts)
        X_train_all = X_all[:n_tr]
        X_test_all = X_all[n_tr:]
    del all_raw_texts, X_all
    for _, tr in feature_extractor.transformer_list:
        if hasattr(tr, "stop_words_"):
            tr.stop_words_ = None
    del feature_extractor
    gc.collect()"""

assert target_step0 in content, "target_step0 not found"
content = content.replace(target_step0, repl_step0, 1)

target_stage_gates = """        # ── 1. DeBERTa-v3-large full fine-tune (GPU 0) ──
        if deberta_dir is not None and ((time.perf_counter() - pipeline_t0) / 3600.0 < MAX_WALLCLOCK_HOURS - 1.0):
            try:
                print(f"\\n[Fold {fold+1}] DeBERTa-v3-large full fine-tune ({deb_dev}, BS=10, GA=2, L=352)", flush=True)
                val_deb, test_deb = train_deberta_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    deberta_dir, LABELS,
                    device=deb_dev,
                    batch_size=10, grad_accum=2, max_length=352,
                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                )
                if val_deb is not None:
                    oof_deb[va_idx] = val_deb
                    test_deb_folds.append(test_deb)
            except Exception as e:
                import traceback
                print(f"  [DeBERTa FT] Fold {fold+1} error: {e}", flush=True)
                traceback.print_exc()

        gc.collect()
        if cuda_ok:
            for gi in range(n_gpus):
                with torch.cuda.device(gi):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        # ── 2. BGE-large full fine-tune (GPU 1) ──
        if bge_dir is not None and ((time.perf_counter() - pipeline_t0) / 3600.0 < MAX_WALLCLOCK_HOURS - 0.8):
            try:
                print(f"\\n[Fold {fold+1}] BGE-large full fine-tune ({bge_dev}, BS=12, GA=2, L=352)", flush=True)
                val_bge, test_bge = train_bge_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    bge_dir, LABELS,
                    device=bge_dev,
                    batch_size=12, grad_accum=2, max_length=352,
                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                )
                if val_bge is not None:
                    oof_bge[va_idx] = val_bge
                    test_bge_folds.append(test_bge)
            except Exception as e:
                import traceback
                print(f"  [BGE FT] Fold {fold+1} error: {e}", flush=True)
                traceback.print_exc()

        gc.collect()
        if cuda_ok:
            for gi in range(n_gpus):
                with torch.cuda.device(gi):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        # ── 3. Qwen 2.5 7B LoRA fine-tune (Dual GPU) ───────────
        if qwen_dir is not None and ((time.perf_counter() - pipeline_t0) / 3600.0 < MAX_WALLCLOCK_HOURS - 0.6):
            try:
                print(f"\\n[Fold {fold+1}] Qwen 2.5 7B LoRA fine-tune (device_map=auto, r=64, BS=8, epochs={N_EPOCHS_QWEN}, L=352)", flush=True)
                val_qwen, test_qwen = train_qwen_lora_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    qwen_dir, LABELS,
                    batch_size=8, grad_accum=2, max_length=352, epochs=N_EPOCHS_QWEN,
                    lr_lora=1.2e-4, lr_head=3.0e-4, lora_r=64, lora_alpha=128,
                    deadline=pipeline_deadline,
                )
                if val_qwen is not None:
                    oof_qwen[va_idx] = val_qwen
                    test_qwen_folds.append(test_qwen)
            except Exception as e:
                import traceback
                print(f"  [Qwen LoRA] Fold {fold+1} error: {e}", flush=True)
                traceback.print_exc()"""

repl_stage_gates = """        # ── 1. DeBERTa-v3-large full fine-tune (GPU 0) ──
        deb_expected = np.mean(model_durations["deb"]) * 1.15 if model_durations["deb"] else 600.0
        if deberta_dir is not None and (time.perf_counter() + deb_expected + SAFETY_MARGIN_SEC < pipeline_deadline):
            try:
                print(f"\\n[Fold {fold+1}] DeBERTa-v3-large full fine-tune ({deb_dev}, BS=8, GA=2, L=352)", flush=True)
                m_t0 = time.perf_counter()
                val_deb, test_deb = train_deberta_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    deberta_dir, LABELS,
                    device=deb_dev,
                    batch_size=8, grad_accum=2, max_length=352,
                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                )
                if val_deb is not None and np.all(np.isfinite(val_deb)):
                    oof_deb[va_idx] = val_deb
                    test_deb_folds.append(test_deb)
                    model_durations["deb"].append(time.perf_counter() - m_t0)
            except Exception as e:
                import traceback
                print(f"  [DeBERTa FT] Fold {fold+1} error: {e}", flush=True)
                traceback.print_exc()

        gc.collect()
        if cuda_ok:
            for gi in range(n_gpus):
                with torch.cuda.device(gi):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        # ── 2. BGE-large full fine-tune (GPU 1) ──
        bge_expected = np.mean(model_durations["bge"]) * 1.15 if model_durations["bge"] else 450.0
        if bge_dir is not None and (time.perf_counter() + bge_expected + SAFETY_MARGIN_SEC < pipeline_deadline):
            try:
                print(f"\\n[Fold {fold+1}] BGE-large full fine-tune ({bge_dev}, BS=12, GA=2, L=352)", flush=True)
                m_t0 = time.perf_counter()
                val_bge, test_bge = train_bge_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    bge_dir, LABELS,
                    device=bge_dev,
                    batch_size=12, grad_accum=2, max_length=352,
                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                )
                if val_bge is not None and np.all(np.isfinite(val_bge)):
                    oof_bge[va_idx] = val_bge
                    test_bge_folds.append(test_bge)
                    model_durations["bge"].append(time.perf_counter() - m_t0)
            except Exception as e:
                import traceback
                print(f"  [BGE FT] Fold {fold+1} error: {e}", flush=True)
                traceback.print_exc()

        gc.collect()
        if cuda_ok:
            for gi in range(n_gpus):
                with torch.cuda.device(gi):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        # ── 3. Qwen 2.5 7B LoRA fine-tune (Dual GPU) ───────────
        qwen_expected = np.mean(model_durations["qwen"]) * 1.15 if model_durations["qwen"] else 2700.0
        if qwen_dir is not None and (time.perf_counter() + qwen_expected + SAFETY_MARGIN_SEC < pipeline_deadline):
            try:
                print(f"\\n[Fold {fold+1}] Qwen 2.5 7B LoRA fine-tune (device_map=auto, r=64, BS=8, epochs={N_EPOCHS_QWEN}, L=352)", flush=True)
                m_t0 = time.perf_counter()
                val_qwen, test_qwen = train_qwen_lora_fold(
                    texts_tr, y_tr_str, texts_va, y_va_str, texts_test,
                    qwen_dir, LABELS,
                    batch_size=8, grad_accum=2, max_length=352, epochs=N_EPOCHS_QWEN,
                    lr_lora=1.2e-4, lr_head=3.0e-4, lora_r=64, lora_alpha=128,
                    deadline=pipeline_deadline,
                )
                if val_qwen is not None and np.all(np.isfinite(val_qwen)):
                    oof_qwen[va_idx] = val_qwen
                    test_qwen_folds.append(test_qwen)
                    model_durations["qwen"].append(time.perf_counter() - m_t0)
            except Exception as e:
                import traceback
                print(f"  [Qwen LoRA] Fold {fold+1} error: {e}", flush=True)
                traceback.print_exc()"""

assert target_stage_gates in content, "target_stage_gates not found"
content = content.replace(target_stage_gates, repl_stage_gates, 1)

# Add model_durations tracker definition before fold loop
target_loop_init = "    pipeline_deadline = pipeline_t0 + (MAX_WALLCLOCK_HOURS * 3600) - SAFETY_MARGIN_SEC"
repl_loop_init = """    pipeline_deadline = pipeline_t0 + (MAX_WALLCLOCK_HOURS * 3600) - SAFETY_MARGIN_SEC
    model_durations = {"deb": [], "bge": [], "qwen": []}"""
assert target_loop_init in content, "target_loop_init not found"
content = content.replace(target_loop_init, repl_loop_init, 1)

# Update Diagnostic Ridge Stacker to 5-Fold cross_val_predict
target_diag = """    # Diagnostic: Ridge Stacker Comparison
    try:
        from sklearn.linear_model import RidgeClassifier
        avail_streams = [margins_to_probs(oof_svm, temp=0.20)]
        if oof_deb is not None: avail_streams.append(margins_to_probs(oof_deb, temp=1.2))
        if oof_qwen is not None: avail_streams.append(margins_to_probs(oof_qwen, temp=1.5))
        if oof_bge is not None: avail_streams.append(margins_to_probs(oof_bge, temp=1.2))
        if len(avail_streams) > 1:
            stack_mask = ~np.isnan(avail_streams[0][:, 0])
            for st in avail_streams[1:]:
                stack_mask = stack_mask & ~np.isnan(st[:, 0])
            if np.any(stack_mask):
                lbl2id = {l: i for i, l in enumerate(LABELS)}
                y_stack = np.array([lbl2id[y] for y in y_true[stack_mask]])
                X_oof_stack = np.hstack([st[stack_mask] for st in avail_streams])
                ridge_stacker = RidgeClassifier(alpha=1.0)
                ridge_stacker.fit(X_oof_stack, y_stack)
                ridge_preds = ridge_stacker.predict(X_oof_stack)
                ridge_score = balanced_accuracy_score(y_stack, ridge_preds)
                print(f"  OOF RIDGE STACKER: {ridge_score*100:.2f}% (diagnostic comparison)\\n", flush=True)
    except Exception as e:
        print(f"  [Diagnostic Stacker] Note: {e}", flush=True)"""

repl_diag = """    # Diagnostic: Ridge Stacker Comparison (5-Fold CV out-of-sample)
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
        print(f"  [Diagnostic Stacker] Note: {e}", flush=True)"""

assert target_diag in content, "target_diag not found"
content = content.replace(target_diag, repl_diag, 1)

# Garbage Stream Rejection
target_qwen_mask = """    if oof_qwen is not None and np.any(~np.isnan(oof_qwen[:, 0])):
        qwen_mask = ~np.isnan(oof_qwen[:, 0])
        qwen_score = balanced_accuracy_score(y_true[qwen_mask], [LABELS[i] for i in np.argmax(oof_qwen[qwen_mask], axis=1)])
        print(f"  Qwen LoRA:       {qwen_score*100:.2f}% (folds evaluated: {np.sum(qwen_mask)//max(n_train//n_splits, 1)}/{n_splits})", flush=True)
    else:
        oof_qwen = None
        print("  Qwen LoRA:       (skipped)", flush=True)"""

repl_qwen_mask = """    if oof_qwen is not None and np.any(~np.isnan(oof_qwen).any(axis=1)):
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
        print("  Qwen LoRA:       (skipped)", flush=True)"""

assert target_qwen_mask in content, "target_qwen_mask not found"
content = content.replace(target_qwen_mask, repl_qwen_mask, 1)

# Write out the hardened code
path.write_text(content, encoding="utf-8")
print("Pass 3 of audit fixes applied successfully!")

