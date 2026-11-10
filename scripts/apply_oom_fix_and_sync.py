import ast
import re
from pathlib import Path

def apply_fix():
    ens_path = Path("notebooks/kaggle_tri_ensemble.py")
    dry_path = Path("notebooks/kaggle_dry_run.py")

    with open(ens_path, "r", encoding="utf-8") as f:
        ens_code = f.read()

    with open(dry_path, "r", encoding="utf-8") as f:
        dry_code = f.read()

    marker_ens = "def run_quad_ensemble():"
    marker_dry = "def run_dry_run_verifier():"

    ens_idx = ens_code.index(marker_ens)
    dry_tail = dry_code[dry_code.index(marker_dry):]

    # --- 1. In ens_code shared section:
    # Set enable_grad_checkpointing=True by default in train_transformer_fold
    ens_shared = ens_code[:ens_idx]

    # Replace enable_grad_checkpointing=False, in train_transformer_fold signature
    trans_sig_old = '''def train_transformer_fold(
    model_name,
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    device="cuda:0",
    batch_size=10,
    grad_accum=2,
    max_length=352,
    epochs=4,
    lr_backbone=1.5e-5,
    lr_head=1.2e-4,
    weight_decay=0.01,
    label_smoothing=0.05,
    enable_grad_checkpointing=False,'''

    trans_sig_new = '''def train_transformer_fold(
    model_name,
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    device="cuda:0",
    batch_size=8,
    grad_accum=2,
    max_length=352,
    epochs=4,
    lr_backbone=1.5e-5,
    lr_head=1.2e-4,
    weight_decay=0.01,
    label_smoothing=0.05,
    enable_grad_checkpointing=True,'''

    assert trans_sig_old in ens_shared, "Could not find trans_sig_old in ens_shared!"
    ens_shared = ens_shared.replace(trans_sig_old, trans_sig_new, 1)

    # In train_deberta_fold and train_bge_fold:
    wrappers_old = '''def train_deberta_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs):
    kwargs.setdefault("batch_size", 16)
    kwargs.setdefault("grad_accum", 1)
    return train_transformer_fold(
        "DeBERTa FT", texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs
    )

# --- 7. BGE-LARGE FULL FINE-TUNE WRAPPER (GPU 1) ---

def train_bge_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs):
    kwargs.setdefault("batch_size", 24)
    kwargs.setdefault("grad_accum", 1)
    return train_transformer_fold(
        "BGE FT", texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs
    )'''

    wrappers_new = '''def train_deberta_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs):
    kwargs.setdefault("batch_size", 8)
    kwargs.setdefault("grad_accum", 2)
    kwargs.setdefault("enable_grad_checkpointing", True)
    return train_transformer_fold(
        "DeBERTa FT", texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs
    )

# --- 7. BGE-LARGE FULL FINE-TUNE WRAPPER (GPU 1) ---

def train_bge_fold(texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs):
    kwargs.setdefault("batch_size", 16)
    kwargs.setdefault("grad_accum", 1)
    kwargs.setdefault("enable_grad_checkpointing", True)
    return train_transformer_fold(
        "BGE FT", texts_tr, y_tr_str, texts_va, y_va_str, texts_te, model_dir, labels, **kwargs
    )'''

    assert wrappers_old in ens_shared, "Could not find wrappers_old in ens_shared!"
    ens_shared = ens_shared.replace(wrappers_old, wrappers_new, 1)

    # In train_qwen_lora_fold signature:
    qwen_sig_old = '''def train_qwen_lora_fold(
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    batch_size=12,
    grad_accum=2,
    max_length=352,'''

    qwen_sig_new = '''def train_qwen_lora_fold(
    texts_tr, y_tr_str, texts_va, y_va_str, texts_te,
    model_dir, labels,
    batch_size=8,
    grad_accum=2,
    max_length=352,'''

    assert qwen_sig_old in ens_shared, "Could not find qwen_sig_old in ens_shared!"
    ens_shared = ens_shared.replace(qwen_sig_old, qwen_sig_new, 1)

    # --- 2. In ens_code unshared section (run_quad_ensemble):
    ens_tail = ens_code[ens_idx:]

    # Update batch sizes in run_quad_ensemble
    ens_tail = ens_tail.replace(
        'print(f"\\n[Fold {fold+1}] Launching CONCURRENT Dual-GPU Training: DeBERTa ({deb_dev}, BS=16) || BGE ({bge_dev}, BS=24)", flush=True)',
        'print(f"\\n[Fold {fold+1}] Launching CONCURRENT Dual-GPU Training: DeBERTa ({deb_dev}, BS=8, GA=2) || BGE ({bge_dev}, BS=16, GA=1)", flush=True)'
    )
    ens_tail = ens_tail.replace('batch_size=16, grad_accum=1, max_length=352,\n                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,',
                                'batch_size=8, grad_accum=2, max_length=352,\n                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=True,')
    ens_tail = ens_tail.replace('batch_size=24, grad_accum=1, max_length=352,\n                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,',
                                'batch_size=16, grad_accum=1, max_length=352,\n                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=True,')
    ens_tail = ens_tail.replace('batch_size=12, grad_accum=2, max_length=352, epochs=N_EPOCHS_QWEN,',
                                'batch_size=8, grad_accum=2, max_length=352, epochs=N_EPOCHS_QWEN,')
    ens_tail = ens_tail.replace('DeBERTa-v3-large full fine-tune ({deb_dev}, BS=16, GA=1, L=352)',
                                'DeBERTa-v3-large full fine-tune ({deb_dev}, BS=8, GA=2, L=352)')
    ens_tail = ens_tail.replace('BGE-large full fine-tune ({bge_dev}, BS=24, GA=1, L=352)',
                                'BGE-large full fine-tune ({bge_dev}, BS=16, GA=1, L=352)')
    ens_tail = ens_tail.replace('Qwen 2.5 7B LoRA fine-tune (device_map=auto, r=32, BS=12, GA=2,',
                                'Qwen 2.5 7B LoRA fine-tune (device_map=auto, r=32, BS=8, GA=2,')

    # --- 3. In dry_tail (run_dry_run_verifier):
    dry_tail = dry_tail.replace('batch_size=16, grad_accum=1, max_length=352, epochs=1,\n            enable_grad_checkpointing=False,',
                                'batch_size=8, grad_accum=2, max_length=352, epochs=1,\n            enable_grad_checkpointing=True,')
    dry_tail = dry_tail.replace('batch_size=24, grad_accum=1, max_length=352, epochs=1,\n            enable_grad_checkpointing=False,',
                                'batch_size=16, grad_accum=1, max_length=352, epochs=1,\n            enable_grad_checkpointing=True,')
    dry_tail = dry_tail.replace('batch_size=12, grad_accum=2, max_length=352, epochs=1,',
                                'batch_size=8, grad_accum=2, max_length=352, epochs=1,')

    # Full files
    final_ens_code = ens_shared + ens_tail
    final_dry_code = ens_shared + dry_tail

    with open(ens_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(final_ens_code)

    with open(dry_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(final_dry_code)

    print("Successfully wrote upgraded files.")
    ast.parse(final_ens_code)
    print("kaggle_tri_ensemble.py AST is valid.")
    ast.parse(final_dry_code)
    print("kaggle_dry_run.py AST is valid.")

if __name__ == "__main__":
    apply_fix()

