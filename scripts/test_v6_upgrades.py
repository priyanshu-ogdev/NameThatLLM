import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from notebooks.kaggle_tri_ensemble import (
    compute_family_logits,
    SpikeAndGapHierarchicalHead,
    CPUGradientMetaCombiner,
    NUM_CLASSES,
    LABELS,
    FAMILIES,
    FAMILY_TO_INDICES,
)

def test_v6_upgrades():
    print("Testing V6 Architectural Upgrades...")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Test Temperature-Smoothed Family Logits
    logits = torch.randn(10, NUM_CLASSES)
    # Set one sibling logit high for Google (indices 5, 6, 7, 8)
    logits[:, 5] = 12.0  # gemini-3.1-pro-preview
    logits[:, 6] = 8.0   # gemini-3.6-flash
    logits[:, 7] = 6.0   # gemini-3.7-flash
    logits[:, 8] = 4.0   # gemini-3.8-flash

    fam_logits_smooth = compute_family_logits(logits, temp=4.0)
    assert fam_logits_smooth.shape == (10, len(FAMILIES))
    assert not torch.isnan(fam_logits_smooth).any()

    # Check gradient flow to non-top siblings
    logits.requires_grad = True
    fam_smooth = compute_family_logits(logits, temp=4.0)
    loss = fam_smooth[:, 1].sum()  # Google family is index 1
    loss.backward()
    assert logits.grad[:, 5].abs().mean() > 0.0
    assert logits.grad[:, 6].abs().mean() > 0.0  # non-top sibling receives non-zero gradient!
    assert logits.grad[:, 7].abs().mean() > 0.0
    assert logits.grad[:, 8].abs().mean() > 0.0
    print("  [Pass] Temperature-smoothed compute_family_logits distributes gradients across all siblings.")

    # 2. Test Multi-Layer Fusion in SpikeAndGapHierarchicalHead
    head = SpikeAndGapHierarchicalHead(hidden_size=64, num_classes=NUM_CLASSES, sub_centers=3)
    B, T, D = 4, 16, 64
    mask = torch.ones((B, T), dtype=torch.long)
    mask[:, 10:] = 0

    # Test with tuple of layers [-1, -5, -9]
    fake_layers = [torch.randn(B, T, D) for _ in range(12)]
    logits_tuple = head(fake_layers, mask)
    assert logits_tuple.shape == (B, NUM_CLASSES)
    loss_tuple = logits_tuple.sum()
    loss_tuple.backward()
    assert head.layer_mix_weights.grad is not None
    assert not torch.isnan(head.layer_mix_weights.grad).any()
    print("  [Pass] Multi-layer hidden state fusion in SpikeAndGapHierarchicalHead backward verified.")

    # Test backward compatibility with single tensor
    head.zero_grad()
    single_layer = torch.randn(B, T, D)
    logits_single = head(single_layer, mask)
    assert logits_single.shape == (B, NUM_CLASSES)
    loss_single = logits_single.sum()
    loss_single.backward()
    print("  [Pass] SpikeAndGapHierarchicalHead backward compatibility with single tensor verified.")

    # 3. Test fit_nested_cv in CPUGradientMetaCombiner
    N = 250
    y_synth = np.random.randint(0, NUM_CLASSES, N)
    groups = np.repeat(np.arange(50), 5)  # 50 prompt groups
    mock_oof = [np.random.randn(N, NUM_CLASSES) for _ in range(4)]
    for i in range(N):
        mock_oof[0][i, y_synth[i]] += 1.0
        mock_oof[1][i, y_synth[i]] += 1.5
        mock_oof[2][i, y_synth[i]] += 2.0
        mock_oof[3][i, y_synth[i]] += 0.8

    combiner = CPUGradientMetaCombiner(n_models=4, n_classes=NUM_CLASSES, l2_reg=0.05)
    cv_probs, cv_score = combiner.fit_nested_cv(
        mock_oof, y_synth, groups=groups, prior_weights=[0.25, 0.20, 0.45, 0.10], n_splits=5, max_iter=25
    )
    assert cv_probs.shape == (N, NUM_CLASSES)
    assert np.all(cv_probs >= 0.0)
    assert np.allclose(cv_probs.sum(axis=1), 1.0)
    assert 0.0 <= cv_score <= 1.0
    print(f"  [Pass] CPUGradientMetaCombiner fit_nested_cv 5-Fold Grouped CV verified (cv_score={cv_score:.4f}).")

    # Test bagged parameters on predict_proba
    test_probs = combiner.predict_proba([m[:10] for m in mock_oof])
    assert test_probs.shape == (10, NUM_CLASSES)
    assert np.allclose(test_probs.sum(axis=1), 1.0)
    print("  [Pass] CPUGradientMetaCombiner bagged parameters predict_proba verified.")

    print("\n>>> ALL V6 UPGRADE VERIFICATIONS PASSED [OK] <<<")

if __name__ == "__main__":
    test_v6_upgrades()

