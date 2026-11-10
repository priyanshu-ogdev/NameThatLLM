import re
import sys
from pathlib import Path

path = Path("notebooks/kaggle_tri_ensemble.py")
content = path.read_text(encoding="utf-8")

# ==============================================================================
# 1. Update module docstring & imports
# ==============================================================================
target_header = '''"""
Name That LLM: Quad-Model Frontier Ensemble (v3 - Max Hardware Utilization)
DeBERTa-v3-large Full FT (GPU 0) + BGE-large Full FT (GPU 1) + Qwen 7B LoRA (Dual GPU) + CPU Dual-Linear (1.2M)
Standalone, offline, self-contained Kaggle submission notebook.

Target Score: 0.75+ Macro Balanced Accuracy (18-class LLM authorship attribution)

Hardware Utilization Design:
- Accelerator: GPU T4 x2 (Dual GPU, 2x15 GB VRAM = 30 GB total)
- Internet: Off (Strict Kaggle offline competition rule)
- VRAM Optimization:
  1. DeBERTa-v3-large FT -> Dedicated to cuda:0 (~8.0 GB / 15 GB VRAM)
  2. BGE-large-en-v1.5 FT -> Dedicated to cuda:1 (~8.2 GB / 15 GB VRAM)
  3. Qwen 2.5 7B LoRA   -> device_map="auto" across both GPUs (~21 GB / 30 GB VRAM, max_length=512)
- RAM Optimization:
  4. CPU Dual-Linear   -> 1,200,000 sparse TF-IDF features + 115 stylometrics (LinearSVC + Ridge)
                         uses ~6 GB of Kaggle's 30 GB System RAM.
- Quad Borda+z blend    -> Hierarchical Family Gating -> Sinkhorn-Knopp balancing -> submission.parquet
"""'''

repl_header = '''"""
Name That LLM: Quad-Model Frontier Ensemble (v4 - Hardened Production Pipeline)
DeBERTa-v3-large Full FT (GPU 0) + BGE-large Full FT (GPU 1) + Qwen 7B LoRA (Dual GPU) + CPU Dual-Linear (~650k)
Standalone, offline, self-contained Kaggle submission notebook.

Target Score: 0.75+ Macro Balanced Accuracy (18-class LLM authorship attribution)

Architecture & Hardware Design:
- Accelerator: GPU T4 x2 (Dual Tesla T4, 2x14.56 GB = 29.12 GB total VRAM)
- Internet: Off (Strict Kaggle offline competition rule)
- Hardware Isolation:
  1. DeBERTa-v3-large FT -> Dedicated to cuda:0 (BS=8, Dynamic Padding + Smart Mega-Batching, Head+Tail L=352)
  2. BGE-large-en-v1.5 FT -> Dedicated to cuda:1 (BS=12, Dynamic Padding + Smart Mega-Batching, Head+Tail L=352)
  3. Qwen 2.5 7B LoRA   -> device_map="auto" sharded across dual GPUs (r=64, alpha=128, ~160M trainable params)
  4. CPU Dual-Linear   -> ~650,000 sparse TF-IDF features + 134 stylometrics (LinearSVC + Ridge)
- Consensus Engine:
  Calibrated Probability Consensus (T_svm=0.20) -> Cardinality-Normalized Family Gating -> Sinkhorn-Knopp Balancing
"""'''

assert target_header in content, "target_header not found"
content = content.replace(target_header, repl_header, 1)

# Clean dead imports
for dead in [
    "import scipy\n",
    "from scipy.stats import rankdata\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
]:
    content = content.replace(dead, "")

# ==============================================================================
# 2. Update RichStyleFeatures (PUNCTUATION & CONTRACTION_RE)
# ==============================================================================
target_punct = """    PUNCTUATION = [
        "\\u2014", "\\u2013", "-", ",", ".", ";", ":", "!", "?", "\\u2026", "...",
        '"', "'", "\\u2018", "\\u201c", "\\u201d", "(", ")", "[", "]",
        "*", "#", "\\n", "\\t", "/", "%", "`",
    ]
    FUNCTION_WORDS = (
        "the a an and or but if then so because although however therefore thus "
        "moreover furthermore indeed perhaps might could would should may must can will "
        "we you i it they this that these those not no very more most such"
    ).split()
    CONTRACTION_RE = re.compile(r"\\b\\w+['’\\u2019]\\w+\\b")
    WORD_RE = re.compile(r"\\b\\w+(?:['’\\u2019]\\w+)?\\b")"""

repl_punct = """    PUNCTUATION = [
        "\\u2014", "\\u2013", "-", ",", ".", ";", ":", "!", "?", "\\u2026", "...",
        '"', "'", "\\u2018", "\\u2019", "\\u201c", "\\u201d", "(", ")", "[", "]",
        "*", "#", "\\n", "\\t", "/", "%", "`",
    ]
    FUNCTION_WORDS = (
        "the a an and or but if then so because although however therefore thus "
        "moreover furthermore indeed perhaps might could would should may must can will "
        "we you i it they this that these those not no very more most such"
    ).split()
    CONTRACTION_RE = re.compile(r"\\b\\w+['\\u2019\\u2018]\\w+\\b")
    WORD_RE = re.compile(r"\\b\\w+(?:['\\u2019\\u2018]\\w+)?\\b")"""

assert target_punct in content, "target_punct not found"
content = content.replace(target_punct, repl_punct, 1)

# ==============================================================================
# 3. Update tokenize_with_head_tail to support preserve_newlines flag
# ==============================================================================
target_tokenize = """def tokenize_with_head_tail(tokenizer, texts, max_length=352, head_tokens=224, tail_tokens=128):
    \"\"\"
    Tokenize sequences with Head+Tail truncation.
    Preserves opening thesis and concluding sign-off signatures where LLM stylistic patterns peak.
    Returns unpadded token lists ready for dynamic micro-batch padding.
    \"\"\"
    prep_texts = [prep_transformer_text(t) for t in texts]
    enc = tokenizer(prep_texts, truncation=False, padding=False)"""

repl_tokenize = """def tokenize_with_head_tail(tokenizer, texts, max_length=352, head_tokens=224, tail_tokens=128, preserve_newlines=True):
    \"\"\"
    Tokenize sequences with Head+Tail truncation.
    Preserves opening thesis and concluding sign-off signatures where LLM stylistic patterns peak.
    For encoder models (DeBERTa/BGE), preserve_newlines maps newlines to dedicated [NL]/[PAR] tokens.
    For decoder models (Qwen), raw newlines are preserved natively.
    Returns unpadded token lists ready for dynamic micro-batch padding.
    \"\"\"
    if preserve_newlines:
        prep_texts = [prep_transformer_text(t) for t in texts]
    else:
        prep_texts = [t if isinstance(t, str) else "" for t in texts]
    enc = tokenizer(prep_texts, truncation=False, padding=False)"""

assert target_tokenize in content, "target_tokenize not found"
content = content.replace(target_tokenize, repl_tokenize, 1)

# ==============================================================================
# 4. Update train_transformer_fold
# ==============================================================================
target_tr_body = """    model.config.use_cache = False
    in_features = model.classifier.in_features if hasattr(model.classifier, "in_features") else model.config.hidden_size
    model.classifier = MultiSampleDropoutHead(in_features, len(labels))

    if enable_grad_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    model = model.to(dev)
    ckpt_status = "ON" if enable_grad_checkpointing else "OFF (high throughput)"
    print(f"  [{model_name}] Ready on {dev} (FP16 Tensor Cores ON, grad_ckpt={ckpt_status}, Multi-Sample Dropout ON). {get_mem_status(dev)}", flush=True)

    cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.25)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    opt_params = get_optimizer_grouped_parameters(
        model, lr_head=lr_head, lr_backbone=lr_backbone, layer_decay=0.92, weight_decay=weight_decay
    )
    optimizer = AdamW(opt_params)

    n_opt_steps_per_epoch = max((len(texts_tr) + batch_size * grad_accum - 1) // (batch_size * grad_accum), 1)
    total_steps = n_opt_steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(0.1 * total_steps), 1),
        num_training_steps=total_steps,
    )

    print(f"  [{model_name}] Tokenizing {len(texts_tr)} train / {len(texts_va)} val / {len(texts_te)} test passages (Head+Tail L={max_length}, [NL] structure preserved)...", flush=True)
    enc_tr = tokenize_with_head_tail(tokenizer, texts_tr, max_length=max_length)
    enc_va = tokenize_with_head_tail(tokenizer, texts_va, max_length=max_length)
    enc_te = tokenize_with_head_tail(tokenizer, texts_te, max_length=max_length)

    use_amp = cuda_ok
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def infer_val():
        if len(texts_va) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        model.eval()
        all_logits = []
        eval_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_va), eval_bs):
                end = min(start + eval_bs, len(texts_va))
                sl = {
                    "input_ids": enc_va["input_ids"][start:end],
                    "attention_mask": enc_va["attention_mask"][start:end],
                }
                if "token_type_ids" in enc_va:
                    sl["token_type_ids"] = enc_va["token_type_ids"][start:end]
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                kw = {
                    "input_ids": padded["input_ids"].to(dev, non_blocking=True),
                    "attention_mask": padded["attention_mask"].to(dev, non_blocking=True),
                }
                if "token_type_ids" in padded:
                    kw["token_type_ids"] = padded["token_type_ids"].to(dev, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits = model(**kw).logits
                all_logits.append(logits.float().cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    def infer_test():
        if len(texts_te) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        model.eval()
        all_logits = []
        eval_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_te), eval_bs):
                end = min(start + eval_bs, len(texts_te))
                sl = {
                    "input_ids": enc_te["input_ids"][start:end],
                    "attention_mask": enc_te["attention_mask"][start:end],
                }
                if "token_type_ids" in enc_te:
                    sl["token_type_ids"] = enc_te["token_type_ids"][start:end]
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                kw = {
                    "input_ids": padded["input_ids"].to(dev, non_blocking=True),
                    "attention_mask": padded["attention_mask"].to(dev, non_blocking=True),
                }
                if "token_type_ids" in padded:
                    kw["token_type_ids"] = padded["token_type_ids"].to(dev, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits = model(**kw).logits
                all_logits.append(logits.float().cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    best_val_score = -1.0
    best_val_logits = None
    best_state = None
    tr_lengths = np.array([len(x) for x in enc_tr["input_ids"]])
    n_total_micros = (len(texts_tr) + batch_size - 1) // batch_size
    mega_size = batch_size * 5

    for epoch in range(epochs):
        model.train()
        # Smart Mega-Batching: Shuffle in chunks of 50, then sort by length within each chunk
        # Reduces attention padding overhead by ~40% while preserving stochastic SGD diversity
        perm = np.random.permutation(len(texts_tr))
        sorted_indices = []
        for m_start in range(0, len(texts_tr), mega_size):
            m_chunk = perm[m_start:m_start + mega_size]
            sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
        epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad()
        running_loss = 0.0
        n_micro = 0
        accumulated_steps = 0

        for micro_step, start in enumerate(range(0, len(texts_tr), batch_size)):
            idx = epoch_indices[start:start + batch_size]
            batch_slice = {
                "input_ids": [enc_tr["input_ids"][j] for j in idx],
                "attention_mask": [enc_tr["attention_mask"][j] for j in idx],
            }
            if "token_type_ids" in enc_tr:
                batch_slice["token_type_ids"] = [enc_tr["token_type_ids"][j] for j in idx]

            padded_micro = tokenizer.pad(batch_slice, padding=True, return_tensors="pt")
            batch_kwargs = {
                "input_ids": padded_micro["input_ids"].to(dev, non_blocking=True),
                "attention_mask": padded_micro["attention_mask"].to(dev, non_blocking=True),
            }
            if "token_type_ids" in padded_micro:
                batch_kwargs["token_type_ids"] = padded_micro["token_type_ids"].to(dev, non_blocking=True)
            batch_labels = torch.tensor(y_tr_int[idx], dtype=torch.long, device=dev)

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

            accumulated_steps += 1
            running_loss += loss.item() * grad_accum
            n_micro += 1
            last_micro = (start + batch_size) >= len(texts_tr)

            if accumulated_steps >= grad_accum or (last_micro and accumulated_steps > 0):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulated_steps = 0

            if (n_micro % 30 == 0) or last_micro:
                mem_s = get_mem_status(dev)
                print(f"    [{model_name} E{epoch+1} Step {n_micro}/{n_total_micros}] loss={loss.item()*grad_accum:.4f} | {mem_s}", flush=True)

        avg_loss = running_loss / max(n_micro, 1)

        val_logits = infer_val()
        val_loss = float(nn.CrossEntropyLoss()(
            torch.tensor(val_logits, dtype=torch.float32),
            torch.tensor(y_va_int, dtype=torch.long),
        ).item())
        val_preds = np.argmax(val_logits, axis=1)
        val_bal_acc = float(balanced_accuracy_score(y_va_int, val_preds))
        mem_s = get_mem_status(dev)
        print(
            f"  [{model_name}] E{epoch+1}/{epochs} | "
            f"train={avg_loss:.4f} val={val_loss:.4f} bal_acc={val_bal_acc*100:.2f}% | {mem_s}",
            flush=True,
        )

        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_val_logits = val_logits.copy()
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  [{model_name}] New best epoch {epoch+1} (bal_acc={val_bal_acc*100:.2f}%)", flush=True)

        if deadline is not None and time.perf_counter() > deadline:
            print(f"  [{model_name}] [Time Guard] Deadline reached after epoch {epoch+1}/{epochs}. Stopping early to preserve predictions.", flush=True)
            break

        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    best_test_logits = infer_test()

    del model, tokenizer, optimizer, scheduler, weight_tensor, enc_tr, enc_va, enc_te, best_state
    gc.collect()
    if cuda_ok:
        for gi in range(n_gpus):
            with torch.cuda.device(gi):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    print(f"  [{model_name}] Done. best_bal_acc={best_val_score*100:.2f}%", flush=True)
    return best_val_logits, best_test_logits"""

repl_tr_body = """    # Add dedicated visible tokens for newlines and resize embeddings
    num_added = tokenizer.add_tokens(["[NL]", "[PAR]"], special_tokens=True)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    in_features = model.classifier.in_features if hasattr(model.classifier, "in_features") else model.config.hidden_size
    model.classifier = MultiSampleDropoutHead(in_features, len(labels))

    if enable_grad_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    model = model.to(dev)
    ckpt_status = "ON" if enable_grad_checkpointing else "OFF (auto OOM fallback)"
    print(f"  [{model_name}] Ready on {dev} (FP16 Tensor Cores ON, grad_ckpt={ckpt_status}, Multi-Sample Dropout ON). {get_mem_status(dev)}", flush=True)

    cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.25)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    opt_params = get_optimizer_grouped_parameters(
        model, lr_head=lr_head, lr_backbone=lr_backbone, layer_decay=0.92, weight_decay=weight_decay
    )
    optimizer = AdamW(opt_params)

    n_opt_steps_per_epoch = max((len(texts_tr) + batch_size * grad_accum - 1) // (batch_size * grad_accum), 1)
    total_steps = n_opt_steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(0.1 * total_steps), 1),
        num_training_steps=total_steps,
    )

    print(f"  [{model_name}] Tokenizing {len(texts_tr)} train / {len(texts_va)} val / {len(texts_te)} test passages (Head+Tail L={max_length}, [NL] structure preserved)...", flush=True)
    enc_tr = tokenize_with_head_tail(tokenizer, texts_tr, max_length=max_length, preserve_newlines=True)
    enc_va = tokenize_with_head_tail(tokenizer, texts_va, max_length=max_length, preserve_newlines=True)
    enc_te = tokenize_with_head_tail(tokenizer, texts_te, max_length=max_length, preserve_newlines=True)

    use_amp = cuda_ok
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def infer_val():
        if len(texts_va) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        model.eval()
        va_lengths = [len(x) for x in enc_va["input_ids"]]
        sort_idx = np.argsort(va_lengths)
        all_logits = []
        eval_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_va), eval_bs):
                end = min(start + eval_bs, len(texts_va))
                batch_indices = sort_idx[start:end]
                sl = {
                    "input_ids": [enc_va["input_ids"][j] for j in batch_indices],
                    "attention_mask": [enc_va["attention_mask"][j] for j in batch_indices],
                }
                if "token_type_ids" in enc_va:
                    sl["token_type_ids"] = [enc_va["token_type_ids"][j] for j in batch_indices]
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                kw = {
                    "input_ids": padded["input_ids"].to(dev, non_blocking=True),
                    "attention_mask": padded["attention_mask"].to(dev, non_blocking=True),
                }
                if "token_type_ids" in padded:
                    kw["token_type_ids"] = padded["token_type_ids"].to(dev, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits = model(**kw).logits
                all_logits.append(logits.float().cpu().numpy())
        sorted_logits = np.concatenate(all_logits, axis=0)
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(len(sort_idx))
        res = sorted_logits[inv_idx]
        return np.nan_to_num(res, nan=0.0)

    def infer_test():
        if len(texts_te) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        model.eval()
        te_lengths = [len(x) for x in enc_te["input_ids"]]
        sort_idx = np.argsort(te_lengths)
        all_logits = []
        eval_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_te), eval_bs):
                end = min(start + eval_bs, len(texts_te))
                batch_indices = sort_idx[start:end]
                sl = {
                    "input_ids": [enc_te["input_ids"][j] for j in batch_indices],
                    "attention_mask": [enc_te["attention_mask"][j] for j in batch_indices],
                }
                if "token_type_ids" in enc_te:
                    sl["token_type_ids"] = [enc_te["token_type_ids"][j] for j in batch_indices]
                padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
                kw = {
                    "input_ids": padded["input_ids"].to(dev, non_blocking=True),
                    "attention_mask": padded["attention_mask"].to(dev, non_blocking=True),
                }
                if "token_type_ids" in padded:
                    kw["token_type_ids"] = padded["token_type_ids"].to(dev, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits = model(**kw).logits
                all_logits.append(logits.float().cpu().numpy())
        sorted_logits = np.concatenate(all_logits, axis=0)
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(len(sort_idx))
        res = sorted_logits[inv_idx]
        return np.nan_to_num(res, nan=0.0)

    best_val_score = -1.0
    best_val_logits = None
    best_state = None
    tr_lengths = np.array([len(x) for x in enc_tr["input_ids"]])
    n_total_micros = (len(texts_tr) + batch_size - 1) // batch_size
    mega_size = batch_size * 5
    stop_early = False

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(texts_tr))
        sorted_indices = []
        for m_start in range(0, len(texts_tr), mega_size):
            m_chunk = perm[m_start:m_start + mega_size]
            sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
        epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad()
        running_loss = 0.0
        n_micro = 0
        accumulated_steps = 0

        for micro_step, start in enumerate(range(0, len(texts_tr), batch_size)):
            if deadline is not None and (micro_step % 10 == 0) and time.perf_counter() > deadline:
                print(f"  [{model_name}] [Time Guard] Deadline reached mid-epoch at step {micro_step}/{n_total_micros}. Stopping training.", flush=True)
                stop_early = True
                break

            idx = epoch_indices[start:start + batch_size]
            batch_slice = {
                "input_ids": [enc_tr["input_ids"][j] for j in idx],
                "attention_mask": [enc_tr["attention_mask"][j] for j in idx],
            }
            if "token_type_ids" in enc_tr:
                batch_slice["token_type_ids"] = [enc_tr["token_type_ids"][j] for j in idx]

            padded_micro = tokenizer.pad(batch_slice, padding=True, return_tensors="pt")
            batch_kwargs = {
                "input_ids": padded_micro["input_ids"].to(dev, non_blocking=True),
                "attention_mask": padded_micro["attention_mask"].to(dev, non_blocking=True),
            }
            if "token_type_ids" in padded_micro:
                batch_kwargs["token_type_ids"] = padded_micro["token_type_ids"].to(dev, non_blocking=True)
            batch_labels = torch.tensor(y_tr_int[idx], dtype=torch.long, device=dev)

            try:
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
                    loss.backward()

            accumulated_steps += 1
            running_loss += loss.item() * grad_accum
            n_micro += 1
            last_micro = (start + batch_size) >= len(texts_tr)

            if accumulated_steps >= grad_accum or (last_micro and accumulated_steps > 0):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = scaler.get_scale()
                    if new_scale >= old_scale:
                        scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad()
                accumulated_steps = 0

            if (n_micro % 30 == 0) or last_micro:
                mem_s = get_mem_status(dev)
                print(f"    [{model_name} E{epoch+1} Step {n_micro}/{n_total_micros}] loss={loss.item()*grad_accum:.4f} | {mem_s}", flush=True)

        avg_loss = running_loss / max(n_micro, 1)

        val_logits = infer_val()
        val_loss = float(nn.CrossEntropyLoss()(
            torch.tensor(val_logits, dtype=torch.float32),
            torch.tensor(y_va_int, dtype=torch.long),
        ).item())
        val_preds = np.argmax(val_logits, axis=1)
        val_bal_acc = float(balanced_accuracy_score(y_va_int, val_preds))
        mem_s = get_mem_status(dev)
        print(
            f"  [{model_name}] E{epoch+1}/{epochs} | "
            f"train={avg_loss:.4f} val={val_loss:.4f} bal_acc={val_bal_acc*100:.2f}% | {mem_s}",
            flush=True,
        )

        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_val_logits = val_logits.copy()
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  [{model_name}] New best epoch {epoch+1} (bal_acc={val_bal_acc*100:.2f}%)", flush=True)

        if stop_early or (deadline is not None and time.perf_counter() > deadline):
            print(f"  [{model_name}] [Time Guard] Stopping early to preserve predictions.", flush=True)
            break

        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    best_test_logits = infer_test()

    n_gpus_local = torch.cuda.device_count() if cuda_ok else 0
    del model, tokenizer, optimizer, scheduler, weight_tensor, enc_tr, enc_va, enc_te, best_state
    gc.collect()
    if cuda_ok:
        for gi in range(n_gpus_local):
            with torch.cuda.device(gi):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    print(f"  [{model_name}] Done. best_bal_acc={best_val_score*100:.2f}%", flush=True)
    return best_val_logits, best_test_logits"""

assert target_tr_body in content, "target_tr_body not found"
content = content.replace(target_tr_body, repl_tr_body, 1)

path.write_text(content, encoding="utf-8")
print("Pass 2 of audit fixes applied!")
