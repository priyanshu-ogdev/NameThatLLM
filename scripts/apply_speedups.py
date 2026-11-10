import re

# 1. Read kaggle_tri_ensemble.py
with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    text = f.read()

# Add inference_context right below amp_autocast
inf_def = """def amp_autocast():
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()

def inference_context():
    \"\"\"Inference context with PyTorch inference_mode for maximum dispatch throughput.\"\"\"
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()
"""
assert 'def amp_autocast():\n    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):\n        return torch.amp.autocast("cuda")\n    return torch.cuda.amp.autocast()\n' in text, "Could not find amp_autocast"
text = text.replace(
    'def amp_autocast():\n    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):\n        return torch.amp.autocast("cuda")\n    return torch.cuda.amp.autocast()\n',
    inf_def
)

# Replace torch.no_grad with inference_context in evaluate_transformer_split
old_eval_trans = """    cuda_ok = torch.cuda.is_available() and getattr(dev, "type", str(dev)) != "cpu"
    with torch.no_grad():
        for start in range(0, n_samples, eval_bs):"""
new_eval_trans = """    cuda_ok = torch.cuda.is_available() and getattr(dev, "type", str(dev)) != "cpu"
    with inference_context():
        for start in range(0, n_samples, eval_bs):"""
assert old_eval_trans in text, "Could not find evaluate_transformer_split torch.no_grad"
text = text.replace(old_eval_trans, new_eval_trans)

# Replace in evaluate_qwen_split
old_qwen_eval = """    inf_bs = 24
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
                "input_ids": padded["input_ids"].pin_memory().to(first_dev, non_blocking=True) if cuda_ok else padded["input_ids"].to(first_dev),
                "attention_mask": padded["attention_mask"].pin_memory().to(first_dev, non_blocking=True) if cuda_ok else padded["attention_mask"].to(first_dev),
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
            all_logits.append(logits)"""

new_qwen_eval = """    inf_bs = 24
    with inference_context():
        for start in range(0, n_samples, inf_bs):
            end = min(start + inf_bs, n_samples)
            batch_indices = sort_idx[start:end]
            sl = {
                "input_ids": [enc_data["input_ids"][j] for j in batch_indices],
                "attention_mask": [enc_data["attention_mask"][j] for j in batch_indices],
            }
            padded = tokenizer.pad(sl, padding=True, return_tensors="pt")
            batch_kwargs = {
                "input_ids": padded["input_ids"].pin_memory().to(first_dev, non_blocking=True) if cuda_ok else padded["input_ids"].to(first_dev),
                "attention_mask": padded["attention_mask"].pin_memory().to(first_dev, non_blocking=True) if cuda_ok else padded["attention_mask"].to(first_dev),
            }
            if cuda_ok:
                with amp_autocast():
                    out = backbone(**batch_kwargs, output_hidden_states=False)
            else:
                out = backbone(**batch_kwargs, output_hidden_states=False)

            mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device)
            sum_hidden = (out.last_hidden_state * mask).sum(dim=1).float()
            pooled = sum_hidden / torch.clamp(mask.sum(dim=1).float(), min=1e-9)
            logits = head(pooled.to(head_dev)).float().cpu().numpy()
            all_logits.append(logits)"""
assert old_qwen_eval in text, "Could not find evaluate_qwen_split block"
text = text.replace(old_qwen_eval, new_qwen_eval)

# Optimize pooling in train_qwen_lora_fold
old_qwen_train_pool = """                    mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
                    hidden_f = out.last_hidden_state.float()
                    pooled = (hidden_f * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                    logits = head(pooled.to(head_dev))
                    loss = criterion(logits.float(), batch_labels) / grad_accum"""

new_qwen_train_pool = """                    mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device)
                    sum_hidden = (out.last_hidden_state * mask).sum(dim=1).float()
                    pooled = sum_hidden / torch.clamp(mask.sum(dim=1).float(), min=1e-9)
                    logits = head(pooled.to(head_dev))
                    loss = criterion(logits.float(), batch_labels) / grad_accum"""
assert old_qwen_train_pool in text, "Could not find train_qwen_lora_fold pooling"
text = text.replace(old_qwen_train_pool, new_qwen_train_pool)

# Optimize state_dict copy in train_transformer_fold
old_best_state = "best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}"
new_best_state = "best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}"
assert old_best_state in text, "Could not find best_state copy"
text = text.replace(old_best_state, new_best_state)

# Increase mega_size to batch_size * 8 in train_transformer_fold and train_qwen_lora_fold
old_mega = "mega_size = batch_size * 5"
new_mega = "mega_size = batch_size * 8"
assert text.count(old_mega) == 2, f"Expected 2 occurrences of {old_mega}, found {text.count(old_mega)}"
text = text.replace(old_mega, new_mega)

with open("notebooks/kaggle_tri_ensemble.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated notebooks/kaggle_tri_ensemble.py with all speedup optimizations successfully!")

# Now sync shared code with kaggle_dry_run.py
marker_ens = "def run_quad_ensemble():"
marker_dry = "def run_dry_run_verifier():"

with open("notebooks/kaggle_dry_run.py", "r", encoding="utf-8") as f:
    dry_text = f.read()

shared_code = text[:text.index(marker_ens)]
dry_tail = dry_text[dry_text.index(marker_dry):]

new_dry_text = shared_code + dry_tail

with open("notebooks/kaggle_dry_run.py", "w", encoding="utf-8") as f:
    f.write(new_dry_text)

print("Synchronized shared code into notebooks/kaggle_dry_run.py successfully!")

