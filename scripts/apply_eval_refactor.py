import re

def refactor_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Remove duplicate `import time` at line 88 (near Counter, gc, etc.)
    content = content.replace("import re\nimport time\nimport warnings", "import re\nimport warnings")

    # 2. Add evaluate_transformer_split and evaluate_qwen_split right before train_transformer_fold
    eval_funcs = '''def evaluate_transformer_split(model, enc_data, tokenizer, dev, use_amp, num_classes):
    """
    Length-sorted batched inference for encoder models.
    Groups evaluation passages by length to eliminate redundant padding tokens,
    then inverts the sort order back to the original index sequence.
    """
    n_samples = len(enc_data["input_ids"]) if "input_ids" in enc_data else 0
    if n_samples == 0:
        return np.zeros((0, num_classes), dtype=np.float64)
    model.eval()
    lengths = [len(x) for x in enc_data["input_ids"]]
    sort_idx = np.argsort(lengths)
    all_logits = []
    eval_bs = 32
    with torch.no_grad():
        for start in range(0, n_samples, eval_bs):
            end = min(start + eval_bs, n_samples)
            batch_indices = sort_idx[start:end]
            sl = {
                "input_ids": [enc_data["input_ids"][j] for j in batch_indices],
                "attention_mask": [enc_data["attention_mask"][j] for j in batch_indices],
            }
            if "token_type_ids" in enc_data:
                sl["token_type_ids"] = [enc_data["token_type_ids"][j] for j in batch_indices]
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


def evaluate_qwen_split(backbone, head, enc_data, tokenizer, first_dev, head_dev, cuda_ok, num_classes):
    """
    Length-sorted batched inference for Qwen 2.5 7B LoRA decoder.
    Passes representations through backbone to MultiSampleDropoutHead.
    """
    n_samples = len(enc_data["input_ids"]) if "input_ids" in enc_data else 0
    if n_samples == 0:
        return np.zeros((0, num_classes), dtype=np.float64)
    backbone.eval()
    head.eval()
    lengths = [len(x) for x in enc_data["input_ids"]]
    sort_idx = np.argsort(lengths)
    all_logits = []
    inf_bs = 32
    with torch.no_grad():
        for start in range(0, n_samples, inf_bs):
            end = min(start + inf_bs, n_samples)
            batch_indices = sort_idx[start:end]
            sl = {
                "input_ids": [enc_data["input_ids"][j] for j in batch_indices],
                "attention_mask": [enc_data["attention_mask"][j] for j in batch_indices],
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


def train_transformer_fold('''

    content = content.replace("def train_transformer_fold(", eval_funcs)

    # 3. In train_transformer_fold: replace infer_val and infer_test definitions with direct calls
    # Find block between scaler = torch.cuda.amp.GradScaler(enabled=use_amp) and best_val_score = -1.0
    transformer_infer_block = re.search(r'scaler = torch\.cuda\.amp\.GradScaler\(enabled=use_amp\)\n\n    def infer_val\(\):.*?best_val_score = -1\.0', content, re.DOTALL)
    if transformer_infer_block:
        replacement = "scaler = torch.cuda.amp.GradScaler(enabled=use_amp)\n\n    best_val_score = -1.0"
        content = content[:transformer_infer_block.start()] + replacement + content[transformer_infer_block.end():]
    else:
        print(f"Warning: could not find transformer_infer_block in {path}")

    # In train_transformer_fold: replace calls to infer_val() and infer_test()
    content = content.replace("val_logits = infer_val()", "val_logits = evaluate_transformer_split(model, enc_va, tokenizer, dev, use_amp, len(labels))", 1)
    content = content.replace("best_test_logits = infer_test()", "best_test_logits = evaluate_transformer_split(model, enc_te, tokenizer, dev, use_amp, len(labels))", 1)

    # 4. In train_qwen_lora_fold: replace infer_val and infer_test definitions with direct calls
    qwen_infer_block = re.search(r'scaler = torch\.cuda\.amp\.GradScaler\(enabled=cuda_ok\)\n\n    def infer_val\(\):.*?best_val_score = -1\.0', content, re.DOTALL)
    if qwen_infer_block:
        replacement = "scaler = torch.cuda.amp.GradScaler(enabled=cuda_ok)\n\n    best_val_score = -1.0"
        content = content[:qwen_infer_block.start()] + replacement + content[qwen_infer_block.end():]
    else:
        print(f"Warning: could not find qwen_infer_block in {path}")

    # In train_qwen_lora_fold: replace calls to infer_val() and infer_test()
    content = content.replace("val_logits = infer_val()", "val_logits = evaluate_qwen_split(backbone, head, enc_va, tokenizer, first_dev, head_dev, cuda_ok, len(labels))", 1)
    content = content.replace("best_test_logits = infer_test()", "best_test_logits = evaluate_qwen_split(backbone, head, enc_te, tokenizer, first_dev, head_dev, cuda_ok, len(labels))", 1)

    # 5. Fix string formatting and items() -> values()
    content = content.replace(
        'print(f"  [Qwen LoRA] [Time Guard] Stopping early to preserve predictions.", flush=True)',
        'print("  [Qwen LoRA] [Time Guard] Stopping early to preserve predictions.", flush=True)'
    )
    content = content.replace(
        'for fam, indices in FAMILY_TO_INDICES.items():',
        'for indices in FAMILY_TO_INDICES.values():'
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully refactored {path}")

refactor_file("notebooks/kaggle_tri_ensemble.py")
refactor_file("notebooks/kaggle_dry_run.py")

