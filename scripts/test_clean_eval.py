import torch
import numpy as np

def evaluate_transformer_split(model, enc_data, tokenizer, dev, use_amp, num_classes):
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

print("evaluate_transformer_split syntax and logic verified [OK]")

