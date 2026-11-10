import os
import glob
import pandas as pd
import numpy as np

print("=" * 70)
print("AUDITING ALL DATASETS: PROMPT STRUCTURE, TEST COLUMNS & CLASS COUNTS")
print("=" * 70)

for path in sorted(glob.glob("data/**/*.csv", recursive=True) + glob.glob("data/**/*.parquet", recursive=True)):
    if "submission" in path:
        continue
    print(f"\nFile: {path}")
    try:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if "model" in df.columns:
            vc = df["model"].value_counts()
            print(f"  Models: {len(vc)} classes | Min count: {vc.min()}, Max count: {vc.max()}, Mean: {vc.mean():.1f}")
            print(f"  Value counts snippet:\n{vc.head(4)}\n  ...\n{vc.tail(4)}")
        if "prompt_id" in df.columns:
            print(f"  Has prompt_id: Yes ({df['prompt_id'].nunique()} unique prompts)")
            if "model" in df.columns:
                c = df.groupby("prompt_id")["model"].value_counts().unstack(fill_value=0)
                desc = c.describe().T[["min", "mean", "max"]]
                print("  Prompt x Model matrix stats (min, mean, max rows per prompt per model):")
                print(desc)
        else:
            print("  Has prompt_id: NO")
    except Exception as e:
        print(f"  Error loading: {e}")

