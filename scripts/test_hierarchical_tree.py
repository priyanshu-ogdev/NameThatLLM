import sys
sys.path.insert(0, ".")

import numpy as np
import pytest
from notebooks.kaggle_tri_ensemble import (
    compute_hierarchical_tree_probs,
    quad_rank_blend,
    FAMILY_TO_INDICES,
    LABELS,
    NUM_CLASSES,
)

def test_hierarchical_tree_basic():
    np.random.seed(42)
    N = 50
    p_svm = np.random.dirichlet(np.ones(NUM_CLASSES), size=N)
    p_deb = np.random.dirichlet(np.ones(NUM_CLASSES), size=N)
    p_qwen = np.random.dirichlet(np.ones(NUM_CLASSES), size=N)
    p_bge = np.random.dirichlet(np.ones(NUM_CLASSES), size=N)

    # 1. Full 4-stream blend with tree active
    p_tree = compute_hierarchical_tree_probs(
        p_svm, p_deb, p_qwen, p_bge,
        router_weights=(0.30, 0.25, 0.30, 0.15),
        expert_weights=(0.20, 0.15, 0.60, 0.05),
        hierarchy_alpha=0.65,
    )
    assert p_tree.shape == (N, NUM_CLASSES)
    assert np.all(p_tree >= 0.0)
    assert np.allclose(np.sum(p_tree, axis=1), 1.0)

    # 2. Alpha = 0.0 must exactly match flat expert blend
    p_flat = compute_hierarchical_tree_probs(
        p_svm, p_deb, p_qwen, p_bge,
        router_weights=(0.30, 0.25, 0.30, 0.15),
        expert_weights=(0.20, 0.15, 0.60, 0.05),
        hierarchy_alpha=0.0,
    )
    w_e = np.array([0.20, 0.15, 0.60, 0.05])
    w_e /= w_e.sum()
    expected_flat = w_e[0]*p_svm + w_e[1]*p_deb + w_e[2]*p_qwen + w_e[3]*p_bge
    expected_flat /= np.sum(expected_flat, axis=1, keepdims=True)
    assert np.allclose(p_flat, expected_flat)

    # 3. Missing stream resilience (e.g. only SVM and Qwen available)
    p_partial = compute_hierarchical_tree_probs(
        p_svm, None, p_qwen, None,
        router_weights=(0.30, 0.25, 0.30, 0.15),
        expert_weights=(0.20, 0.15, 0.60, 0.05),
        hierarchy_alpha=0.65,
    )
    assert p_partial.shape == (N, NUM_CLASSES)
    assert np.all(p_partial >= 0.0)
    assert np.allclose(np.sum(p_partial, axis=1), 1.0)

    # 4. quad_rank_blend integration test
    svm_m = np.random.randn(N, NUM_CLASSES)
    qwen_m = np.random.randn(N, NUM_CLASSES)
    preds = quad_rank_blend(
        svm_m, None, qwen_m, None,
        weights=(1.0, 0.0, 1.0, 0.0),
        temperatures=(0.20, 1.2, 1.5, 1.2),
        balance_test=True,
        apply_family_gating=True,
        family_alpha=0.65,
    )
    assert len(preds) == N
    assert np.all((preds >= 0) & (preds < NUM_CLASSES))

    print("ALL HIERARCHICAL TREE UNIT TESTS PASSED [OK]!")

if __name__ == "__main__":
    test_hierarchical_tree_basic()
