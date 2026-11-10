import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from notebooks.kaggle_tri_ensemble import (
    find_topic_token_ids,
    TopicBlindAndSiFTPrep,
    StylometricMultiMomentPooler,
    CPUGradientMetaCombiner,
    calibrate_logits,
    NUM_CLASSES,
    LABELS,
)

def test_prep_and_combiner():
    print("Testing TopicBlindAndSiFTPrep, StylometricMultiMomentPooler & CPUGradientMetaCombiner...")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Test find_topic_token_ids
    seqs = [
        [101, 5, 20, 102],
        [101, 5, 30, 102],
        [101, 6, 40, 102],
        [101, 7, 50, 102],
    ]
    groups = np.array([0, 1, 2, 3])
    # Token 5 appears in groups 0 and 1 (2 groups < 3) -> topic token!
    # Tokens 20, 30, 40, 50 appear in 1 group (< 3) -> topic tokens!
    # Special tokens 101, 102 protected
    topic_ids = find_topic_token_ids(seqs, groups, min_prompts=3, special_ids={101, 102})
    assert 101 not in topic_ids and 102 not in topic_ids
    assert 5 in topic_ids and 20 in topic_ids and 30 in topic_ids
    print(f"  [Pass] find_topic_token_ids verified ({len(topic_ids)} topic tokens found, special tokens protected).")

    # 2. Test TopicBlindAndSiFTPrep
    prep = TopicBlindAndSiFTPrep(topic_token_ids=topic_ids, mask_token_id=999, mask_prob=1.0, sift_eps=0.015)
    input_ids = torch.tensor([[101, 5, 20, 102], [101, 6, 7, 102]])
    masked = prep.mask_topic_tokens(input_ids)
    assert masked[0, 1] == 999 and masked[0, 2] == 999
    assert masked[0, 0] == 101 and masked[0, 3] == 102  # specials untouched
    print("  [Pass] TopicBlind stochastic masking verified.")

    # Test SiFT noise
    emb = torch.randn(2, 4, 64, requires_grad=True)
    mask = torch.ones((2, 4), dtype=torch.long)
    noisy_emb = prep.apply_sift_embedding_noise(emb, mask)
    loss = noisy_emb.sum()
    loss.backward()
    assert emb.grad is not None and not torch.isnan(emb.grad).any()
    print("  [Pass] SiFT scale-invariant embedding noise & gradients verified.")

    # 3. Test StylometricMultiMomentPooler
    pooler = StylometricMultiMomentPooler(hidden_size=64)
    h_states = torch.randn(3, 16, 64, requires_grad=True)
    attn_mask = torch.ones((3, 16), dtype=torch.long)
    attn_mask[0, 8:] = 0  # sample 0 has half padding
    pooler.set_mask(attn_mask)
    pooled = pooler(h_states)
    assert pooled.shape == (3, 64)
    assert pooler.output_dim == 64
    p_loss = pooled.sum()
    p_loss.backward()
    assert pooler.dense.weight.grad is not None and not torch.isnan(pooler.dense.weight.grad).any()
    print("  [Pass] StylometricMultiMomentPooler (CLS + Mean + Max) forward & backward verified.")

    # 4. Test calibrate_logits
    raw_l = np.array([[2.0, -1.0], [0.5, 1.5]])
    calib = calibrate_logits(raw_l, temp=2.0)
    assert np.allclose(calib, raw_l / 2.0)
    print("  [Pass] calibrate_logits temperature scaling verified.")

    # 5. Test CPUGradientMetaCombiner imported directly from notebook codebase
    N = 200
    y_synth = np.random.randint(0, NUM_CLASSES, N)
    mock_oof = [np.random.randn(N, NUM_CLASSES) for _ in range(4)]
    for i in range(N):
        mock_oof[0][i, y_synth[i]] += 1.0
        mock_oof[1][i, y_synth[i]] += 1.5
        mock_oof[2][i, y_synth[i]] += 2.0
        mock_oof[3][i, y_synth[i]] += 0.8

    combiner = CPUGradientMetaCombiner(n_models=4, n_classes=NUM_CLASSES, l2_reg=0.05)
    combiner.fit(mock_oof, y_synth, prior_weights=[0.25, 0.20, 0.45, 0.10], max_iter=30)
    probs = combiner.predict_proba(mock_oof)
    assert probs.shape == (N, NUM_CLASSES)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Test CPUGradientMetaCombiner with missing streams [s0, None, s2, s3]
    mock_oof_missing = [mock_oof[0], None, mock_oof[2], mock_oof[3]]
    comb_missing = CPUGradientMetaCombiner(n_models=3, n_classes=NUM_CLASSES, l2_reg=0.05)
    comb_missing.fit(mock_oof_missing, y_synth, prior_weights=[0.25, 0.20, 0.45, 0.10], max_iter=25)
    assert comb_missing.M == 3
    assert comb_missing.valid_indices == [0, 2, 3]
    p_missing = comb_missing.predict_proba(mock_oof_missing)
    assert p_missing.shape == (N, NUM_CLASSES)
    assert np.allclose(p_missing.sum(axis=1), 1.0)
    print("  [Pass] CPUGradientMetaCombiner PyTorch autograd L-BFGS fit & probability simplex verified (including missing streams).")

    print("\n>>> ALL PREP & CPU META-COMBINER TESTS PASSED [OK] <<<")

if __name__ == "__main__":
    test_prep_and_combiner()

