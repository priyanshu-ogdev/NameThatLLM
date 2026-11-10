import re
from pathlib import Path

def patch_tri_ensemble():
    path = Path("notebooks/kaggle_tri_ensemble.py")
    code = path.read_text(encoding="utf-8")

    # 1. Remove AutoConfig import
    old_imp = """from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)"""
    new_imp = """from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)"""
    assert old_imp in code, "Target 1 not found"
    code = code.replace(old_imp, new_imp, 1)

    # 2. Add noqa to StratifiedGroupKFold
    old_sgkf = "from sklearn.model_selection import StratifiedGroupKFold"
    new_sgkf = "from sklearn.model_selection import StratifiedGroupKFold  # noqa: F401"
    assert old_sgkf in code, "Target 2 not found"
    code = code.replace(old_sgkf, new_sgkf, 1)

    # 3. Remove redefinition of AutoModelForSequenceClassification in train_transformer_fold
    old_tf_imp = """    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import get_cosine_schedule_with_warmup"""
    new_tf_imp = """    from transformers import get_cosine_schedule_with_warmup"""
    assert old_tf_imp in code, "Target 3 not found"
    code = code.replace(old_tf_imp, new_tf_imp, 1)

    # 4. Add n_oom_skips in train_transformer_fold
    old_oom_init = """    stop_early = False
    n_skipped_micros = 0
    t_train_start = time.perf_counter()"""
    new_oom_init = """    stop_early = False
    n_skipped_micros = 0
    n_oom_skips = 0
    t_train_start = time.perf_counter()"""
    assert old_oom_init in code, "Target 4 not found"
    code = code.replace(old_oom_init, new_oom_init, 1)

    # 5. Add n_skipped_micros and n_oom_skips in train_qwen_lora_fold
    old_qwen_init = """    t_qwen_train_start = time.perf_counter()
    pure_qwen_sec = 0.0

    for epoch in range(epochs):
        t_qwen_ep_start = time.perf_counter()
        backbone.train()
        head.train()
        perm = np.random.permutation(len(texts_tr))
        sorted_indices = []
        for m_start in range(0, len(texts_tr), mega_size):
            m_chunk = perm[m_start:m_start + mega_size]
            sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
        epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0
        accumulated_steps = 0
        stop_early = False"""

    new_qwen_init = """    t_qwen_train_start = time.perf_counter()
    pure_qwen_sec = 0.0
    stop_early = False
    n_skipped_micros = 0
    n_oom_skips = 0

    for epoch in range(epochs):
        t_qwen_ep_start = time.perf_counter()
        backbone.train()
        head.train()
        perm = np.random.permutation(len(texts_tr))
        sorted_indices = []
        for m_start in range(0, len(texts_tr), mega_size):
            m_chunk = perm[m_start:m_start + mega_size]
            sorted_indices.extend(m_chunk[np.argsort(tr_lengths[m_chunk])])
        epoch_indices = np.array(sorted_indices, dtype=np.int64)

        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0
        accumulated_steps = 0"""
    assert old_qwen_init in code, "Target 5 not found"
    code = code.replace(old_qwen_init, new_qwen_init, 1)

    # 6. Qwen OOM handler guard
    old_qwen_oom = """                except torch.cuda.OutOfMemoryError:
                    print(f"  [Qwen LoRA] CUDA OOM at micro-step {micro_step}. Resetting partial grads...", flush=True)
                    try:
                        del out, pooled, logits, loss
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    if cuda_ok:
                        torch.cuda.empty_cache()
                    continue"""
    new_qwen_oom = """                except torch.cuda.OutOfMemoryError:
                    n_oom_skips += 1
                    print(f"  [Qwen LoRA] CUDA OOM at micro-step {micro_step} (OOM #{n_oom_skips}). Resetting partial grads...", flush=True)
                    if n_oom_skips > 5:
                        raise RuntimeError(f"[Qwen LoRA] Exceeded 5 CUDA OOM errors in fold; aborting to prevent data starvation.")
                    try:
                        del out, pooled, logits, loss
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    if cuda_ok:
                        torch.cuda.empty_cache()
                    continue"""
    assert old_qwen_oom in code, "Target 6 not found"
    code = code.replace(old_qwen_oom, new_qwen_oom, 1)

    path.write_text(code, encoding="utf-8")
    print(f"Successfully patched {path}")

def patch_build_v5_dry_run():
    path = Path("scripts/build_v5_dry_run.py")
    code = path.read_text(encoding="utf-8")

    old_target = """        clf = DualLinearClassifier(c_svc=1.2, alpha_ridge=1.0, svc_weight=0.7, seed=42, canonical_labels=LABELS)
        clf.fit(X_tr, y_tr_str)
        val_svm = clf.decision_function(X_va)
        test_svm = clf.decision_function(X_te)
    cpu_dur = time.perf_counter() - t0"""

    new_target = """        clf = DualLinearClassifier(c_svc=1.2, alpha_ridge=1.0, svc_weight=0.7, seed=42, canonical_labels=LABELS)
        clf.fit(X_tr, y_tr_str)
        val_svm = clf.decision_function(X_va)
        test_svm = clf.decision_function(X_te)
        check_finite_logits(val_svm, "CPU linear val logits")
        check_finite_logits(test_svm, "CPU linear test logits")
    cpu_dur = time.perf_counter() - t0"""

    assert old_target in code, "Dry run build target not found"
    code = code.replace(old_target, new_target, 1)
    path.write_text(code, encoding="utf-8")
    print(f"Successfully patched {path}")

if __name__ == "__main__":
    patch_tri_ensemble()
    patch_build_v5_dry_run()

