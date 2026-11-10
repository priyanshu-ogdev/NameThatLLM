import sys
import os
sys.path.insert(0, ".")
if "NTL_DATA_DIR" not in os.environ:
    os.environ["NTL_DATA_DIR"] = "d:/NameThatLLM/data/sample"
import time
import psutil
import pandas as pd
from notebooks.kaggle_tri_ensemble import build_cpu_feature_extractor, discover_paths

print("Testing memory consumption of build_cpu_feature_extractor...")
vm0 = psutil.virtual_memory().used / (1024**3)

data_dir, *rest = discover_paths()
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")

texts_all = train_df["text"].tolist() + test_df["text"].tolist()
print(f"Total passages to fit: {len(texts_all)}")

t0 = time.time()
fe = build_cpu_feature_extractor(style_weight=0.2)
fe.fit(texts_all)
t_fit = time.time() - t0
print(f"Fitting completed in {t_fit:.1f}s")

t0 = time.time()
X_tr = fe.transform(train_df["text"].tolist())
t_tr = time.time() - t0
print(f"Transform train ({len(train_df)}) in {t_tr:.1f}s, shape={X_tr.shape}, nnz={X_tr.nnz}")

vm1 = psutil.virtual_memory().used / (1024**3)
print(f"RAM used: {vm1 - vm0:.2f} GB")
print(f"Sparse matrix data bytes: {X_tr.data.nbytes / (1024**2):.1f} MB")
