import re

with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    text = f.read()

# Normalize newlines
text = text.replace("\r\n", "\n")

# 1. Update train_transformer_fold signature
sig_target = """    enable_grad_checkpointing=False,
):"""
sig_repl = """    enable_grad_checkpointing=False,
    deadline=None,
):"""
if sig_target in text:
    text = text.replace(sig_target, sig_repl, 1)
    print("1. Added deadline to train_transformer_fold signature.")
else:
    print("Could not find sig_target in text!")

# 2. Add deadline check in train_transformer_fold epoch loop
deb_target = """        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_val_logits = val_logits.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [{model_name}] New best epoch {epoch+1} (bal_acc={val_bal_acc*100:.2f}%)", flush=True)

        model.train()"""

deb_repl = """        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_val_logits = val_logits.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [{model_name}] New best epoch {epoch+1} (bal_acc={val_bal_acc*100:.2f}%)", flush=True)

        if deadline is not None and time.perf_counter() > deadline:
            print(f"  [{model_name}] [Time Guard] Deadline reached after epoch {epoch+1}/{epochs}. Stopping early to preserve predictions.", flush=True)
            break

        model.train()"""

if deb_target in text:
    text = text.replace(deb_target, deb_repl, 1)
    print("2. Added intra-epoch deadline check to train_transformer_fold.")
else:
    print("Could not find deb_target in text!")

# 3. Update train_qwen_lora_fold signature
qwen_sig_target = """    lora_dropout=0.05,
):"""
qwen_sig_repl = """    lora_dropout=0.05,
    deadline=None,
):"""
if qwen_sig_target in text:
    text = text.replace(qwen_sig_target, qwen_sig_repl, 1)
    print("3. Added deadline to train_qwen_lora_fold signature.")
else:
    print("Could not find qwen_sig_target in text!")

# 4. Add deadline check in train_qwen_lora_fold epoch loop
qwen_loop_target = """        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_val_logits = val_logits.copy()
            best_lora_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items() if "lora_" in k}
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            print(f"  [Qwen LoRA] New best epoch {epoch+1} (bal_acc={val_bal_acc*100:.2f}%)", flush=True)"""

qwen_loop_repl = """        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_val_logits = val_logits.copy()
            best_lora_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items() if "lora_" in k}
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            print(f"  [Qwen LoRA] New best epoch {epoch+1} (bal_acc={val_bal_acc*100:.2f}%)", flush=True)

        if deadline is not None and time.perf_counter() > deadline:
            print(f"  [Qwen LoRA] [Time Guard] Deadline reached after epoch {epoch+1}/{epochs}. Stopping early to preserve predictions.", flush=True)
            break"""

if qwen_loop_target in text:
    text = text.replace(qwen_loop_target, qwen_loop_repl, 1)
    print("4. Added intra-epoch deadline check to train_qwen_lora_fold.")
else:
    print("Could not find qwen_loop_target in text!")

# 5. Define pipeline_deadline and pass it to all models in run_quad_ensemble
pipe_target = """    N_EPOCHS_DEB = 4
    N_EPOCHS_BGE = 4
    N_EPOCHS_QWEN = 3
    MAX_WALLCLOCK_HOURS = 8.0
    pipeline_t0 = time.perf_counter()"""

pipe_repl = """    N_EPOCHS_DEB = 4
    N_EPOCHS_BGE = 4
    N_EPOCHS_QWEN = 3
    MAX_WALLCLOCK_HOURS = 8.2
    SAFETY_MARGIN_SEC = 1200  # 20 min buffer for final bagging, Sinkhorn, and submission
    pipeline_t0 = time.perf_counter()
    pipeline_deadline = pipeline_t0 + (MAX_WALLCLOCK_HOURS * 3600) - SAFETY_MARGIN_SEC"""

if pipe_target in text:
    text = text.replace(pipe_target, pipe_repl, 1)
    print("5. Defined pipeline_deadline in run_quad_ensemble.")
else:
    print("Could not find pipe_target in text!")

# 6. Pass deadline to model calls in fold loop
call_deb = """                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                )"""
call_deb_repl = """                    epochs=N_EPOCHS_DEB, lr_backbone=1.5e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                )"""
if call_deb in text:
    text = text.replace(call_deb, call_deb_repl, 1)
    print("6. Passed deadline to DeBERTa.")

call_bge = """                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                )"""
call_bge_repl = """                    epochs=N_EPOCHS_BGE, lr_backbone=2.2e-5, lr_head=1.2e-4, enable_grad_checkpointing=False,
                    deadline=pipeline_deadline,
                )"""
if call_bge in text:
    text = text.replace(call_bge, call_bge_repl, 1)
    print("7. Passed deadline to BGE.")

call_qwen = """                    lr_lora=1.2e-4, lr_head=3.0e-4, lora_r=64, lora_alpha=128,
                )"""
call_qwen_repl = """                    lr_lora=1.2e-4, lr_head=3.0e-4, lora_r=64, lora_alpha=128,
                    deadline=pipeline_deadline,
                )"""
if call_qwen in text:
    text = text.replace(call_qwen, call_qwen_repl, 1)
    print("8. Passed deadline to Qwen.")

with open("notebooks/kaggle_tri_ensemble.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Section 3 upgrades successfully applied to kaggle_tri_ensemble.py!")

