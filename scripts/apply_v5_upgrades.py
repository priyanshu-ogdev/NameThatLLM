import re
from pathlib import Path

def apply_upgrades_to_code(content: str, is_dry_run: bool = False) -> str:
    # 1. Update module docstring and add SCRIPT_T0 at the top
    content = content.replace(
        'Name That LLM: Quad-Model Frontier Ensemble (v4 - Hardened Production Pipeline)',
        'Name That LLM: Quad-Model Frontier Ensemble (v5 - Production Hardened Pipeline & Smoke Test)'
    )
    if "SCRIPT_T0 = time.perf_counter()" not in content:
        content = content.replace(
            "import time\n",
            "import time\n\nSCRIPT_T0 = time.perf_counter()\n"
        )

    # 2. Add check_finite_logits and update evaluate_transformer_split / evaluate_qwen_split
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

    # 3. Embedding layer LLRD update in get_optimizer_grouped_parameters
    old_llrd_emb = """    # Embeddings layer (and relative position embeddings for DeBERTa)
    n_layers = len(layers) if layers is not None else 24
    emb_lr = lr_backbone * (layer_decay ** n_layers)
    emb_params = []"""

    new_llrd_emb = """    # Embeddings layer: word embeddings receive higher LR so newly added [NL], [PAR] tokens can adapt
    n_layers = len(layers) if layers is not None else 24
    emb_lr = max(lr_backbone * (layer_decay ** 4), 8e-6)
    emb_params = []"""
    content = content.replace(old_llrd_emb, new_llrd_emb)

    # 4. Token initialization for [NL] and [PAR] in train_transformer_fold
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

    # 5. AdamW foreach=False to prevent peak memory allocation
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
                # Split micro-batch in two half-size chunks to guarantee recovery
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

    # Initialize n_skipped_micros before the epoch loop in train_transformer_fold
    content = content.replace("stop_early = False\n\n    for epoch in range(epochs):", "stop_early = False\n    n_skipped_micros = 0\n\n    for epoch in range(epochs):")

    # Validate raw logits in train_transformer_fold
    content = content.replace(
        "val_logits = evaluate_transformer_split(model, enc_va, tokenizer, dev, use_amp, len(labels))",
        "val_logits = evaluate_transformer_split(model, enc_va, tokenizer, dev, use_amp, len(labels))\n        check_finite_logits(val_logits, f'{model_name} Val')"
    )
    content = content.replace(
        "best_test_logits = evaluate_transformer_split(model, enc_te, tokenizer, dev, use_amp, len(labels))",
        "best_test_logits = evaluate_transformer_split(model, enc_te, tokenizer, dev, use_amp, len(labels))\n    check_finite_logits(best_test_logits, f'{model_name} Test')"
    )

    # In train_qwen_lora_fold: add skip counter & raw logits validation
    content = content.replace(
        "stop_early = False\n\n    for epoch in range(epochs):",
        "stop_early = False\n    n_skipped_micros = 0\n\n    for epoch in range(epochs):"
    )
    content = content.replace(
        "val_logits = evaluate_qwen_split(backbone, head, enc_va, tokenizer, first_dev, head_dev, cuda_ok, len(labels))",
        "val_logits = evaluate_qwen_split(backbone, head, enc_va, tokenizer, first_dev, head_dev, cuda_ok, len(labels))\n        check_finite_logits(val_logits, 'Qwen Val')"
    )
    content = content.replace(
        "best_test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels))",
        "best_test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels))\n    check_finite_logits(best_test_logits, 'Qwen Test')"
    )

    # In train_qwen_lora_fold: OOM cleanup
    old_qwen_oom = """                    del out, pooled, logits, loss
                    except Exception:
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()"""
    new_qwen_oom = """                    del out, pooled, logits, loss
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    gc.collect()
                    if cuda_ok:
                        for gi in range(n_gpus):
                            with torch.cuda.device(gi):
                                torch.cuda.empty_cache()"""
    content = content.replace(old_qwen_oom, new_qwen_oom)

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
    content = content.replace("def margins_to_probs", fit_temp_func)

    # 8. make_prompt_groups: str(text) defense
    content = content.replace(
        'h = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()',
        'h = hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()'
    )

    return content

print("apply_upgrades_to_code function ready.")

