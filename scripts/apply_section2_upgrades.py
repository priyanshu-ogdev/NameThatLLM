import ast
import re

py_path = 'd:/NameThatLLM/notebooks/kaggle_tri_ensemble.py'
with open(py_path, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Replace duplicate train_bge_fold with thin wrapper
bge_start = '# --- 7. BGE-LARGE FULL FINE-TUNE PER FOLD (GPU 1) ---'
qwen_start = '# --- 8. QWEN 2.5 7B LORA FINE-TUNE PER FOLD (DUAL GPU) ---'

assert bge_start in code, "bge_start not found"
assert qwen_start in code, "qwen_start not found"

parts = code.split(bge_start)
pre_bge = parts[0]
post_bge = parts[1].split(qwen_start)[1]

new_bge = """# --- 7. BGE-LARGE FULL FINE-TUNE WRAPPER (GPU 1) ---

def train_bge_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs):
    return train_transformer_fold(
        "BGE FT", texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs
    )


"""

code = pre_bge + new_bge + qwen_start + post_bge

# 2. Upgrade train_qwen_lora_fold:
# - lr_lora=1.2e-4
# - gamma=0.25
# - tokenize_with_head_tail with dynamic left-padding
old_qwen_sig = '    lr_lora=3.5e-5,'
new_qwen_sig = '    lr_lora=1.2e-4,'
assert old_qwen_sig in code, "old_qwen_sig not found"
code = code.replace(old_qwen_sig, new_qwen_sig, 1)

old_qwen_cw = 'cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.35)'
new_qwen_cw = 'cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.25)'
assert old_qwen_cw in code, "old_qwen_cw not found"
code = code.replace(old_qwen_cw, new_qwen_cw, 1)

# Upgrade Qwen tokenization to dynamic padding
old_qwen_tok = '''    # Pre-tokenize all training texts once on CPU — eliminates all tokenizer GIL contention during GPU training
    print(f"  [Qwen LoRA] Pre-tokenizing {len(texts_tr)} train passages (max_length={max_length})...", flush=True)
    clean_tr = [clean_text(t) for t in texts_tr]
    enc_tr = tokenizer(
        clean_tr, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    tr_input_ids = enc_tr["input_ids"]
    tr_attention_mask = enc_tr["attention_mask"]
    if cuda_ok:
        try:
            tr_input_ids = tr_input_ids.pin_memory()
            tr_attention_mask = tr_attention_mask.pin_memory()
        except Exception:
            pass

    # Pre-tokenize validation set once for zero redundant string processing
    clean_va = [clean_text(t) for t in texts_va]
    enc_va = tokenizer(
        clean_va, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    va_input_ids = enc_va["input_ids"]
    va_attention_mask = enc_va["attention_mask"]
    if cuda_ok:
        try:
            va_input_ids = va_input_ids.pin_memory()
            va_attention_mask = va_attention_mask.pin_memory()
        except Exception:
            pass'''

new_qwen_tok = '''    print(f"  [Qwen LoRA] Tokenizing {len(texts_tr)} train passages (Head+Tail L={max_length}, [NL] structure preserved)...", flush=True)
    enc_tr = tokenize_with_head_tail(tokenizer, texts_tr, max_length=max_length)
    enc_va = tokenize_with_head_tail(tokenizer, texts_va, max_length=max_length)'''

assert old_qwen_tok in code, "old_qwen_tok not found"
code = code.replace(old_qwen_tok, new_qwen_tok, 1)

# Upgrade Qwen infer_val and infer_logits to dynamic padding
old_qwen_infer = '''    def infer_val():
        if len(texts_va) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        backbone.eval()
        head.eval()
        all_logits = []
        inf_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_va), inf_bs):
                batch_kwargs = {
                    "input_ids": va_input_ids[start:start + inf_bs].to(first_dev, non_blocking=True),
                    "attention_mask": va_attention_mask[start:start + inf_bs].to(first_dev, non_blocking=True),
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

    def infer_logits(texts_list):
        if len(texts_list) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        backbone.eval()
        head.eval()
        all_logits = []
        inf_bs = 32
        clean_list = [clean_text(t) for t in texts_list]
        with torch.no_grad():
            for start in range(0, len(clean_list), inf_bs):
                enc = tokenizer(
                    clean_list[start:start + inf_bs], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                )
                batch_kwargs = {
                    "input_ids": enc["input_ids"].to(first_dev, non_blocking=True),
                    "attention_mask": enc["attention_mask"].to(first_dev, non_blocking=True),
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
        return np.concatenate(all_logits, axis=0)'''

new_qwen_infer = '''    def infer_val():
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

    def infer_logits(texts_list):
        if len(texts_list) == 0:
            return np.zeros((0, len(labels)), dtype=np.float64)
        backbone.eval()
        head.eval()
        enc = tokenize_with_head_tail(tokenizer, texts_list, max_length=max_length)
        all_logits = []
        inf_bs = 32
        with torch.no_grad():
            for start in range(0, len(texts_list), inf_bs):
                end = min(start + inf_bs, len(texts_list))
                sl = {
                    "input_ids": enc["input_ids"][start:end],
                    "attention_mask": enc["attention_mask"][start:end],
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
        return np.concatenate(all_logits, axis=0)'''

assert old_qwen_infer in code, "old_qwen_infer not found"
code = code.replace(old_qwen_infer, new_qwen_infer, 1)

# Upgrade Qwen epoch dynamic padding
old_qwen_loop = '''        while i < len(texts_tr):
            idx = perm[i:i + curr_batch]
            batch_kwargs = {
                "input_ids": tr_input_ids[idx].to(first_dev, non_blocking=True),
                "attention_mask": tr_attention_mask[idx].to(first_dev, non_blocking=True),
            }'''

new_qwen_loop = '''        while i < len(texts_tr):
            idx = perm[i:i + curr_batch]
            batch_slice = {
                "input_ids": [enc_tr["input_ids"][j] for j in idx],
                "attention_mask": [enc_tr["attention_mask"][j] for j in idx],
            }
            padded_micro = tokenizer.pad(batch_slice, padding=True, return_tensors="pt")
            batch_kwargs = {
                "input_ids": padded_micro["input_ids"].to(first_dev, non_blocking=True),
                "attention_mask": padded_micro["attention_mask"].to(first_dev, non_blocking=True),
            }'''

assert old_qwen_loop in code, "old_qwen_loop not found"
code = code.replace(old_qwen_loop, new_qwen_loop, 1)

# 3. Update run_quad_ensemble with seed_everything, raw text preservation, and stacker evaluation
old_main_load = '''    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str, "prompt_id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)
    print(f"Train: {len(train_df):,} passages | {train_df['prompt_id'].nunique()} unique prompts", flush=True)
    print(f"Test:  {len(test_df):,} passages\\n", flush=True)

    n_train = len(train_df)
    texts_test = test_df["text"].tolist()
    y_true = train_df["model"].to_numpy()'''

new_main_load = '''    seed_everything(42)

    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str, "prompt_id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})

    # Preserve 100% raw unaltered text for CPU stylometrics & character TF-IDF
    raw_texts_tr = train_df["text"].tolist()
    raw_texts_te = test_df["text"].tolist()

    print(f"Train: {len(train_df):,} passages | {train_df['prompt_id'].nunique()} unique prompts", flush=True)
    print(f"Test:  {len(test_df):,} passages\\n", flush=True)

    n_train = len(train_df)
    texts_test = raw_texts_te
    y_true = train_df["model"].to_numpy()'''

assert old_main_load in code, "old_main_load not found"
code = code.replace(old_main_load, new_main_load, 1)

# In Step 0: fit feature_extractor on raw texts
old_step0_fit = '''    feature_extractor = build_cpu_feature_extractor(style_weight=0.2)
    all_texts = train_df["text"].tolist() + texts_test
    with threadpool_limits(limits=4):
        feature_extractor.fit(all_texts)
        X_train_all = feature_extractor.transform(train_df["text"].tolist())
        X_test_all = feature_extractor.transform(texts_test)'''

new_step0_fit = '''    feature_extractor = build_cpu_feature_extractor(style_weight=0.2)
    all_raw_texts = raw_texts_tr + raw_texts_te
    with threadpool_limits(limits=4):
        feature_extractor.fit(all_raw_texts)
        X_train_all = feature_extractor.transform(raw_texts_tr)
        X_test_all = feature_extractor.transform(raw_texts_te)'''

assert old_step0_fit in code, "old_step0_fit not found"
code = code.replace(old_step0_fit, new_step0_fit, 1)

# In fold loop: seed_everything(42 + fold)
old_fold_start = '''    for fold in range(n_splits):
        fold_start = time.perf_counter()'''

new_fold_start = '''    for fold in range(n_splits):
        seed_everything(42 + fold)
        fold_start = time.perf_counter()'''

assert old_fold_start in code, "old_fold_start not found"
code = code.replace(old_fold_start, new_fold_start, 1)

# In fold loop: pass raw texts to texts_tr and texts_va
old_fold_texts = '''        texts_tr = [train_df["text"].iloc[i] for i in tr_idx]
        texts_va = [train_df["text"].iloc[i] for i in va_idx]'''

new_fold_texts = '''        texts_tr = [raw_texts_tr[i] for i in tr_idx]
        texts_va = [raw_texts_tr[i] for i in va_idx]'''

assert old_fold_texts in code, "old_fold_texts not found"
code = code.replace(old_fold_texts, new_fold_texts, 1)

# In fold loop: update DeBERTa and BGE calls to enable_grad_checkpointing=False
old_deb_call = 'epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4,'
new_deb_call = 'epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,'
assert old_deb_call in code, "old_deb_call not found"
code = code.replace(old_deb_call, new_deb_call, 1)

old_bge_call = 'epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4,'
new_bge_call = 'epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,'
assert old_bge_call in code, "old_bge_call not found"
code = code.replace(old_bge_call, new_bge_call, 1)

old_qwen_call = 'lr_lora=3.5e-5, lr_head=3.0e-4, lora_r=64, lora_alpha=128,'
new_qwen_call = 'lr_lora=1.2e-4, lr_head=3.0e-4, lora_r=64, lora_alpha=128,'
assert old_qwen_call in code, "old_qwen_call not found"
code = code.replace(old_qwen_call, new_qwen_call, 1)

# Verify AST before writing
ast.parse(code)
with open(py_path, 'w', encoding='utf-8') as f:
    f.write(code)

print("All Section 2 upgrades successfully applied and AST validated!")

