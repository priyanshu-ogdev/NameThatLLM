import sys
from pathlib import Path

path = Path("notebooks/kaggle_tri_ensemble.py")
content = path.read_text(encoding="utf-8")

# 1. Remove tautological len(sub) != len(test_ids) check
target_tautology = """    if list(sub.columns) != ["id", "model"]:
        raise ValueError("Submission columns must be exactly ['id', 'model']")
    if len(sub) != len(test_ids):
        raise ValueError(f"Row count mismatch: expected {len(test_ids)}, got {len(sub)}")
    if sub.isna().any().any():"""

repl_tautology = """    if list(sub.columns) != ["id", "model"]:
        raise ValueError("Submission columns must be exactly ['id', 'model']")
    if sub.isna().any().any():"""

assert target_tautology in content, "target_tautology not found"
content = content.replace(target_tautology, repl_tautology, 1)

# 2. Add passage length summary statistics logging at startup
target_stats = """    print(f"Train: {len(train_df):,} passages | {train_df['prompt_id'].nunique()} unique prompts", flush=True)
    print(f"Test:  {len(test_df):,} passages\\n", flush=True)"""

repl_stats = """    tr_lens = train_df["text"].str.len()
    print(f"Train: {len(train_df):,} passages | {train_df['prompt_id'].nunique()} unique prompts (length: mean={tr_lens.mean():.0f}, min={tr_lens.min()}, max={tr_lens.max()} chars)", flush=True)
    print(f"Test:  {len(test_df):,} passages\\n", flush=True)"""

assert target_stats in content, "target_stats not found"
content = content.replace(target_stats, repl_stats, 1)

# 3. Vectorized batch_labels in train_transformer_fold
target_labels = "batch_labels = torch.tensor([y_tr_int[j] for j in idx], dtype=torch.long, device=dev)"
repl_labels = "batch_labels = torch.tensor(y_tr_int[idx], dtype=torch.long, device=dev)"
assert target_labels in content, "target_labels not found"
content = content.replace(target_labels, repl_labels, 1)

path.write_text(content, encoding="utf-8")
print("Successfully applied audit polish to notebooks/kaggle_tri_ensemble.py!")

