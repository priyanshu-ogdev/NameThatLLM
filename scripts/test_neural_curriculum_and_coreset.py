import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import torch
from notebooks.kaggle_tri_ensemble import (
    LABELS,
    NUM_CLASSES,
    select_contrastive_external_coreset,
    sample_epoch_neural_external_slice,
    FastBatchPrefetcher,
    FocalLoss,
    SpikeAndGapHierarchicalHead,
)

def test_coreset_and_curriculum():
    print("=" * 60)
    print("TESTING CONTRASTIVE CORESET & NEURAL CURRICULUM")
    print("=" * 60)

    # 1. Test select_contrastive_external_coreset
    from notebooks.kaggle_tri_ensemble import build_mass_balanced_external_corpus
    train_df = pd.read_csv("data/sample/train.csv")
    ext_path = "datasets/external_short_wide.parquet"
    comb_ext_df, _ = build_mass_balanced_external_corpus(
        train_df, ext_path, LABELS, ext_mass_ratio=0.55, prefer_multi_model=True
    )
    ext_df = comb_ext_df.iloc[len(train_df):].reset_index(drop=True)
    coreset_df = select_contrastive_external_coreset(ext_df, LABELS, max_per_class=240, seed=42)
    assert len(coreset_df) > 0, "Coreset should not be empty"
    counts = coreset_df["model"].value_counts()
    assert (counts <= 240).all(), "Every class capped at 240"
    print(f"  [Pass] Coreset selected {len(coreset_df)} rows across {coreset_df['model'].nunique()} classes (all <= 240/class)")

    # 2. Test sample_epoch_neural_external_slice
    y_ext_int = np.array([LABELS.index(m) for m in coreset_df["model"]], dtype=np.int64)
    group_ids_ext = np.arange(len(coreset_df), dtype=np.int64)
    enc_ext = {
        "input_ids": [[101, 102]] * len(coreset_df),
        "attention_mask": [[1, 1]] * len(coreset_df),
    }

    # Epoch 0 (curriculum representation epoch with 40 samples/class = 720 samples)
    sl_enc, sl_y, sl_g = sample_epoch_neural_external_slice(
        enc_ext, y_ext_int, group_ids_ext, num_classes=18, per_class_k=40, seed=42
    )
    assert len(sl_y) == 18 * 40, f"Expected 720 samples, got {len(sl_y)}"
    unique, counts = np.unique(sl_y, return_counts=True)
    assert len(unique) == 18, "All 18 classes present"
    assert (counts == 40).all(), "Exact uniform 40 samples per class (0.000% prior skew)"
    print("  [Pass] Epoch curriculum slice yields exactly 720 rows (40 per class, 0.000% skew)")

    # 3. Test FastBatchPrefetcher with sample_weights
    class DummyTokenizer:
        def pad(self, sl, padding=True, return_tensors="pt"):
            return {
                "input_ids": torch.tensor(sl["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(sl["attention_mask"], dtype=torch.long),
            }

    dummy_enc = {
        "input_ids": [[10, 20]] * 10,
        "attention_mask": [[1, 1]] * 10,
    }
    dummy_y = np.zeros(10, dtype=np.int64)
    dummy_w = np.array([1.0] * 5 + [0.35] * 5, dtype=np.float32)
    prefetcher = FastBatchPrefetcher(
        DummyTokenizer(), dummy_enc, dummy_y, np.arange(10), batch_size=4, dev="cpu",
        cuda_ok=False, sample_weights=dummy_w,
    )
    weights_seen = []
    for batch_kw, batch_labels in prefetcher:
        assert "_batch_weights" in batch_kw, "Prefetcher must propagate _batch_weights"
        w = batch_kw.pop("_batch_weights")
        weights_seen.extend(w.numpy().tolist())
    prefetcher.close()
    assert np.allclose(weights_seen, dummy_w), "Prefetched weights must match input sample weights"
    print("  [Pass] FastBatchPrefetcher asynchronously delivers pinned sample_weights")

    # 4. Test FocalLoss with domain-discounted sample weights
    crit = FocalLoss(gamma=0.75)
    logits = torch.randn(4, 18, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    weights = torch.tensor([1.0, 1.0, 0.35, 0.35])
    loss = crit(logits, targets, sample_weights=weights)
    assert torch.isfinite(loss), "Loss must be finite"
    loss.backward()
    assert logits.grad is not None and torch.all(torch.isfinite(logits.grad))
    print("  [Pass] FocalLoss computes domain-weighted loss & backprops cleanly")

    print("=" * 60)
    print("ALL CORESET & CURRICULUM UNIT TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_coreset_and_curriculum()
