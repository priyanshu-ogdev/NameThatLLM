import ast

print("Applying audit cleanups...")

with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    ens_text = f.read()

with open("notebooks/kaggle_dry_run.py", "r", encoding="utf-8") as f:
    dry_text = f.read()

# Shared replacements
old_cw_deb = '''    cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.25)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)'''

new_cw_deb = '''    # Balanced 150-sample-per-class competition design: unweighted label-smoothed CE
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)'''

old_cw_qwen = '''    cw = compute_smoothed_class_weights(y_tr_int, num_classes=len(labels), gamma=0.25)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=head_dev)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)'''

new_cw_qwen = '''    # Balanced 150-sample-per-class competition design: unweighted label-smoothed CE
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)'''

for label, target in [
    ("old_cw_deb", old_cw_deb),
    ("old_cw_qwen", old_cw_qwen),
]:
    assert target in ens_text, f"Target {label} not found in ens_text"
    assert target in dry_text, f"Target {label} not found in dry_text"

ens_text = ens_text.replace(old_cw_deb, new_cw_deb, 1)
ens_text = ens_text.replace(old_cw_qwen, new_cw_qwen, 1)

dry_text = dry_text.replace(old_cw_deb, new_cw_deb, 1)
dry_text = dry_text.replace(old_cw_qwen, new_cw_qwen, 1)

# Reset epochs in ens_text to 3, 3, 2 for proper 150-prompt regularization
old_epochs = '''    N_EPOCHS_DEB = 4
    N_EPOCHS_BGE = 4
    N_EPOCHS_QWEN = 3'''

new_epochs = '''    N_EPOCHS_DEB = 3
    N_EPOCHS_BGE = 3
    N_EPOCHS_QWEN = 2'''

assert old_epochs in ens_text, "old_epochs not found in ens_text"
ens_text = ens_text.replace(old_epochs, new_epochs, 1)

# Update dry_run Check 8 formula to 3, 3, 2
old_check8_formula = '''    t_deb_5fold_min = ((n_fold_train * 4 * n_folds) / max(benchmarks["deberta"]["sps"], 0.1)) / 60.0
    t_bge_5fold_min = ((n_fold_train * 4 * n_folds) / max(benchmarks["bge"]["sps"], 0.1)) / 60.0
    t_par_trans_total = max(t_deb_5fold_min, t_bge_5fold_min) if n_gpus >= 2 else (t_deb_5fold_min + t_bge_5fold_min)
    t_qwen_5fold_min = ((n_fold_train * 3 * n_folds) / max(benchmarks["qwen"]["sps"], 0.1)) / 60.0'''

# Check if old_check8_formula or similar exists in dry_text
if "t_deb_total = benchmarks[\"deberta\"][\"5fold_min\"]" in dry_text:
    print("Check 8 uses precomputed benchmarks dictionary.")

ast.parse(ens_text)
ast.parse(dry_text)

with open("notebooks/kaggle_tri_ensemble.py", "w", encoding="utf-8") as f:
    f.write(ens_text)

with open("notebooks/kaggle_dry_run.py", "w", encoding="utf-8") as f:
    f.write(dry_text)

print("Applied audit cleanups successfully!")

