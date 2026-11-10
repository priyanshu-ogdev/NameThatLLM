with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update train_deberta_fold defaults
old_deb_def = """def train_deberta_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs):
    kwargs.setdefault("batch_size", 12)
    kwargs.setdefault("grad_accum", 1)
    kwargs.setdefault("epochs", 3)
    kwargs.setdefault("enable_grad_checkpointing", True)"""

new_deb_def = """def train_deberta_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs):
    kwargs.setdefault("batch_size", 10)
    kwargs.setdefault("grad_accum", 1)
    kwargs.setdefault("epochs", 3)
    kwargs.setdefault("enable_grad_checkpointing", False)"""

assert old_deb_def in text, "Could not find train_deberta_fold defaults"
text = text.replace(old_deb_def, new_deb_def)

# 2. Update train_bge_fold defaults
old_bge_def = """def train_bge_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs):
    kwargs.setdefault("batch_size", 20)
    kwargs.setdefault("grad_accum", 1)
    kwargs.setdefault("epochs", 3)
    kwargs.setdefault("enable_grad_checkpointing", True)"""

new_bge_def = """def train_bge_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs):
    kwargs.setdefault("batch_size", 20)
    kwargs.setdefault("grad_accum", 1)
    kwargs.setdefault("epochs", 3)
    kwargs.setdefault("enable_grad_checkpointing", False)"""

assert old_bge_def in text, "Could not find train_bge_fold defaults"
text = text.replace(old_bge_def, new_bge_def)

# 3. Update train_qwen_lora_fold signature
old_qwen_sig = """def train_qwen_lora_fold(
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    batch_size=8,
    grad_accum=2,"""

new_qwen_sig = """def train_qwen_lora_fold(
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    batch_size=10,
    grad_accum=1,"""

assert old_qwen_sig in text, "Could not find train_qwen_lora_fold signature"
text = text.replace(old_qwen_sig, new_qwen_sig)

# 4. Update callers in run_quad_ensemble
text = text.replace(
    'batch_size=12, grad_accum=1, max_length=352,\n                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=True,',
    'batch_size=10, grad_accum=1, max_length=352,\n                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,'
)

text = text.replace(
    'batch_size=20, grad_accum=1, max_length=352,\n                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=True,',
    'batch_size=20, grad_accum=1, max_length=352,\n                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,'
)

text = text.replace(
    'batch_size=8, grad_accum=2, max_length=352, epochs=N_EPOCHS_QWEN,',
    'batch_size=10, grad_accum=1, max_length=352, epochs=N_EPOCHS_QWEN,'
)

# Also update the print statements in run_quad_ensemble
text = text.replace(
    'DeBERTa ({deb_dev}, BS=12, GA=1) || BGE ({bge_dev}, BS=20, GA=1)',
    'DeBERTa ({deb_dev}, BS=10, GA=1, ckpt=Auto) || BGE ({bge_dev}, BS=20, GA=1, ckpt=Auto)'
)
text = text.replace(
    'Qwen 2.5 7B LoRA fine-tune (device_map=auto, r=32, BS=8, GA=2, epochs={N_EPOCHS_QWEN}, L=352)',
    'Qwen 2.5 7B LoRA fine-tune (device_map=auto, r=32, BS=10, GA=1, epochs={N_EPOCHS_QWEN}, L=352)'
)

with open("notebooks/kaggle_tri_ensemble.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated notebooks/kaggle_tri_ensemble.py successfully!")

# Update kaggle_dry_run.py
with open("notebooks/kaggle_dry_run.py", "r", encoding="utf-8") as f:
    dry_text = f.read()

marker_ens = "def run_quad_ensemble():"
marker_dry = "def run_dry_run_verifier():"

shared_code = text[:text.index(marker_ens)]
dry_tail = dry_text[dry_text.index(marker_dry):]

# Update dry_run calls in dry_tail
dry_tail = dry_tail.replace(
    'batch_size=12, grad_accum=1, max_length=352, epochs=1,\n            enable_grad_checkpointing=True,',
    'batch_size=10, grad_accum=1, max_length=352, epochs=1,\n            enable_grad_checkpointing=False,'
)
dry_tail = dry_tail.replace(
    'batch_size=20, grad_accum=1, max_length=352, epochs=1,\n            enable_grad_checkpointing=True,',
    'batch_size=20, grad_accum=1, max_length=352, epochs=1,\n            enable_grad_checkpointing=False,'
)
dry_tail = dry_tail.replace(
    'batch_size=8, grad_accum=2, max_length=352, epochs=1,',
    'batch_size=10, grad_accum=1, max_length=352, epochs=1,'
)

new_dry_text = shared_code + dry_tail

with open("notebooks/kaggle_dry_run.py", "w", encoding="utf-8") as f:
    f.write(new_dry_text)

print("Synchronized notebooks/kaggle_dry_run.py successfully!")

