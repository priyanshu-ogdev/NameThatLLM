"""
Fixes all 11 ruff F821 / F401 / F841 errors across notebooks/kaggle_tri_ensemble.py and notebooks/kaggle_dry_run.py.
"""
import ast
import subprocess
from pathlib import Path

def fix_errors():
    ens_path = Path("notebooks/kaggle_tri_ensemble.py")
    code = ens_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # 1. Top-level imports for transformers (fixes AutoTokenizer, AutoModel F821)
    old_top_trans = """try:
    import transformers
    transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
except Exception:
    pass"""

    new_top_trans = """from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
try:
    import transformers
    transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
except Exception:
    pass"""

    if old_top_trans in code:
        code = code.replace(old_top_trans, new_top_trans, 1)
    elif "from transformers import AutoTokenizer" not in code:
        code = code.replace("import warnings\n", new_top_trans + "\nimport warnings\n", 1)

    # 2. Fix n_oom_skips scope in train_transformer_fold
    old_trans_fn_head = """def train_transformer_fold(
    model_name: str,
    texts_tr: list, y_tr_str: list,
    texts_va: list, y_va_str: list,
    texts_te: list,
    model_dir: str,
    labels: list,
    batch_size: int = 16,
    grad_accum: int = 1,
    max_length: int = 352,
    epochs: int = 4,
    lr_backbone: float = 2.0e-5,
    lr_head: float = 1.0e-4,
    weight_decay: float = 0.01,
    layer_decay: float = 0.85,
    warmup_ratio: float = 0.10,
    device: str = "cuda:0",
    enable_grad_checkpointing: bool = False,
    deadline: float = None,
):"""

    new_trans_fn_head = """def train_transformer_fold(
    model_name: str,
    texts_tr: list, y_tr_str: list,
    texts_va: list, y_va_str: list,
    texts_te: list,
    model_dir: str,
    labels: list,
    batch_size: int = 16,
    grad_accum: int = 1,
    max_length: int = 352,
    epochs: int = 4,
    lr_backbone: float = 2.0e-5,
    lr_head: float = 1.0e-4,
    weight_decay: float = 0.01,
    layer_decay: float = 0.85,
    warmup_ratio: float = 0.10,
    device: str = "cuda:0",
    enable_grad_checkpointing: bool = False,
    deadline: float = None,
):
    n_oom_skips = 0"""

    if old_trans_fn_head in code:
        code = code.replace(old_trans_fn_head, new_trans_fn_head, 1)

    # 3. Fix n_skipped_micros scope in train_qwen_lora_fold
    old_qwen_fn_head = """def train_qwen_lora_fold(
    texts_tr: list, y_tr_str: list,
    texts_va: list, y_va_str: list,
    texts_te: list,
    model_dir: str,
    labels: list,
    first_dev: str = "cuda:0",
    head_dev: str = "cuda:1",
    batch_size: int = 8,
    grad_accum: int = 2,
    max_length: int = 352,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 64,
    lora_alpha: int = 128,
    deadline: float = None,
):"""

    new_qwen_fn_head = """def train_qwen_lora_fold(
    texts_tr: list, y_tr_str: list,
    texts_va: list, y_va_str: list,
    texts_te: list,
    model_dir: str,
    labels: list,
    first_dev: str = "cuda:0",
    head_dev: str = "cuda:1",
    batch_size: int = 8,
    grad_accum: int = 2,
    max_length: int = 352,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 64,
    lora_alpha: int = 128,
    deadline: float = None,
):
    n_skipped_micros = 0"""

    if old_qwen_fn_head in code:
        code = code.replace(old_qwen_fn_head, new_qwen_fn_head, 1)

    # 4. Fix group_ids_stack in Ridge stacker diagnostic
    old_ridge = "ridge_preds = cross_val_predict(RidgeClassifier(alpha=1.0), X_oof_stack, y_stack, cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), groups=group_ids_stack)"
    new_ridge = """groups_stack = group_ids[stack_mask]
                ridge_preds = cross_val_predict(RidgeClassifier(alpha=1.0), X_oof_stack, y_stack, cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), groups=groups_stack)"""

    if old_ridge in code:
        code = code.replace(old_ridge, new_ridge, 1)

    # Save kaggle_tri_ensemble.py
    ens_path.write_text(code, encoding="utf-8")
    print("Fixed notebooks/kaggle_tri_ensemble.py!")

    # 5. Rebuild kaggle_dry_run.py from shared part of kaggle_tri_ensemble.py
    import sys
    sys.path.insert(0, ".")
    from scripts.build_v5_dry_run import build_v5_dry_run
    build_v5_dry_run()
    print("Rebuilt notebooks/kaggle_dry_run.py!")

    # Fix unused val_svm in dry_run tail if present
    dry_path = Path("notebooks/kaggle_dry_run.py")
    dry_code = dry_path.read_text(encoding="utf-8")
    if "val_svm = " in dry_code and "v_preds = " in dry_code:
        dry_code = dry_code.replace("val_svm = clf.decision_function(X_va)\n", "")
        dry_path.write_text(dry_code, encoding="utf-8")

if __name__ == "__main__":
    fix_errors()

