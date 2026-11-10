import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
from notebooks.kaggle_tri_ensemble import (
    SIBLING_DIFFICULTY_MARGINS,
    FocalLoss,
    NonLinearStylometricKernelExpander,
    DenseStylometricClassifier,
    make_prompt_sibling_paired_indices,
    compute_same_prompt_sibling_repulsion,
    GLOBAL_RAM_MODEL_CACHE,
    FastBatchPrefetcher,
    LABELS,
)

def test_fast_batch_prefetcher_string_groups():
    print("Testing FastBatchPrefetcher with string group_ids...")
    class MockTokenizer:
        def pad(self, batch_slice, padding=True, return_tensors="pt"):
            max_l = max(len(x) for x in batch_slice["input_ids"])
            padded_ids = [x + [0] * (max_l - len(x)) for x in batch_slice["input_ids"]]
            padded_mask = [m + [0] * (max_l - len(m)) for m in batch_slice["attention_mask"]]
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            }

    enc_data = {
        "input_ids": [[101, 1, 2, 102], [101, 3, 102], [101, 4, 5, 6, 102], [101, 7, 102]],
        "attention_mask": [[1, 1, 1, 1], [1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1]],
    }
    y_labels = np.array([0, 1, 2, 3])
    # Pass alphanumeric string prompt IDs (exactly like train.csv prompt_id="p_001")
    string_groups = np.array(["prompt_alpha", "prompt_alpha", "prompt_beta", "prompt_gamma"], dtype=object)
    indices = np.array([0, 1, 2, 3])

    prefetcher = FastBatchPrefetcher(
        tokenizer=MockTokenizer(),
        enc_data=enc_data,
        y_labels=y_labels,
        indices=indices,
        batch_size=2,
        dev="cpu",
        cuda_ok=False,
        group_ids=string_groups,
    )
    batches = list(prefetcher)
    assert len(batches) == 2, f"Expected 2 batches, got {len(batches)}"
    assert prefetcher.current_groups is not None, "Batch groups must be populated"
    assert prefetcher.current_groups.dtype == torch.long, "Batch groups must be torch.long"
    print("  [Pass] FastBatchPrefetcher with string group_ids verified.")



def test_sibling_difficulty_margins_focal_loss():
    print("Testing Sibling Difficulty Margins in FocalLoss...")
    assert len(SIBLING_DIFFICULTY_MARGINS) == 18, "SIBLING_DIFFICULTY_MARGINS length must be 18"
    assert SIBLING_DIFFICULTY_MARGINS[7] == 0.32, "Gemini 3.7 Flash margin must be 0.32"

    criterion = FocalLoss(gamma=1.0, label_smoothing=0.05, use_sibling_margins=True)
    logits = torch.randn(8, 18, requires_grad=True)
    targets = torch.tensor([1, 2, 6, 7, 8, 10, 15, 17], dtype=torch.long)

    loss = criterion(logits, targets)
    assert torch.isfinite(loss), "Loss must be finite"
    loss.backward()
    assert logits.grad is not None and torch.all(torch.isfinite(logits.grad)), "Gradients must be finite"

    # Test dimension compatibility when num_classes != 18
    crit_sub = FocalLoss(use_sibling_margins=True)
    sub_logits = torch.randn(4, 5, requires_grad=True)
    sub_targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    sub_loss = crit_sub(sub_logits, sub_targets)
    assert torch.isfinite(sub_loss), "Must handle non-18 class count gracefully"
    print("  [Pass] Sibling Difficulty Margins in FocalLoss verified.")

def test_nonlinear_stylometric_kernel_expander():
    print("Testing NonLinearStylometricKernelExpander...")
    expander = NonLinearStylometricKernelExpander(n_top_poly=36, n_nystroem=768, seed=42)
    X_toy = np.random.randn(50, 132)
    X_exp = expander.fit_transform(X_toy)

    # 132 (raw) + 36*37/2 (666 quad) + min(768, 49) = 132 + 666 + 49 = 847
    assert X_exp.shape[0] == 50
    assert X_exp.shape[1] == 132 + 666 + 49
    assert np.all(np.isfinite(X_exp)), "Expanded features must be finite"

    # Test transform on new data
    X_new = np.random.randn(10, 132)
    X_new_exp = expander.transform(X_new)
    assert X_new_exp.shape == (10, X_exp.shape[1])
    assert np.all(np.isfinite(X_new_exp)), "Transformed new features must be finite"
    print("  [Pass] NonLinearStylometricKernelExpander verified.")

def test_dense_stylometric_classifier_kernel_expansion():
    print("Testing DenseStylometricClassifier with kernel expansion...")
    clf = DenseStylometricClassifier(c=0.2, seed=42, canonical_labels=LABELS[:6], use_kernel_expander=True)
    X = np.random.randn(40, 132)
    y = np.random.choice(LABELS[:6], size=40)

    clf.fit(X, y)
    probs = clf.predict_proba(X[:5])
    assert probs.shape == (5, 6)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1"
    assert np.all(probs >= 0.0), "Probabilities must be non-negative"
    print("  [Pass] DenseStylometricClassifier with kernel expansion verified.")

def test_prompt_sibling_paired_indices():
    print("Testing make_prompt_sibling_paired_indices...")
    group_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8])
    labels = np.random.randint(0, 18, size=len(group_ids))
    lengths = np.random.randint(50, 250, size=len(group_ids))
    rng = np.random.RandomState(42)

    indices = make_prompt_sibling_paired_indices(group_ids, labels, lengths, batch_size=4, rng=rng)
    assert len(indices) == len(group_ids), "All indices must be retained"
    assert set(indices) == set(range(len(group_ids))), "Permutation must be bijective"
    print("  [Pass] make_prompt_sibling_paired_indices verified.")

def test_same_prompt_sibling_repulsion():
    print("Testing compute_same_prompt_sibling_repulsion...")
    # 4 samples: 0 and 1 have same prompt (group 0) but different models (label 1 vs 2)
    # 2 and 3 have different prompts
    feats = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],  # almost identical to sample 0 -> high cosine similarity collision!
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], requires_grad=True)
    groups = torch.tensor([0, 0, 1, 2])
    labels = torch.tensor([1, 2, 3, 4])

    loss = compute_same_prompt_sibling_repulsion(feats, groups, labels, margin=0.15)
    assert loss.item() > 0.0, "Should penalize same-prompt sibling collision with cos_sim > 0.15"
    loss.backward()
    assert feats.grad is not None and torch.all(torch.isfinite(feats.grad)), "Gradients must be finite"

    # Zero loss case: different prompts or identical model
    diff_groups = torch.tensor([0, 1, 2, 3])
    zero_loss = compute_same_prompt_sibling_repulsion(feats, diff_groups, labels, margin=0.15)
    assert zero_loss.item() == 0.0, "Zero loss when all prompts differ"
    print("  [Pass] compute_same_prompt_sibling_repulsion verified.")

def test_global_ram_model_cache():
    print("Testing GLOBAL_RAM_MODEL_CACHE...")
    GLOBAL_RAM_MODEL_CACHE["test_key"] = torch.randn(10, 10)
    assert "test_key" in GLOBAL_RAM_MODEL_CACHE
    del GLOBAL_RAM_MODEL_CACHE["test_key"]
    print("  [Pass] GLOBAL_RAM_MODEL_CACHE verified.")

def test_hierarchical_tree_ensemble():
    print("Testing hierarchical_tree_ensemble and FAMILY_GROUPS...")
    from notebooks.kaggle_tri_ensemble import hierarchical_tree_ensemble, FAMILY_GROUPS, FAMILY_TO_INDICES, NUM_CLASSES
    assert FAMILY_GROUPS == FAMILY_TO_INDICES, "FAMILY_GROUPS must match FAMILY_TO_INDICES"
    # Verify exact competition indices:
    # OpenAI [0..4], Google [5..8], Anthropic [9..13], Qwen [14], Zhipu [15, 16], Moonshot [17]
    assert FAMILY_GROUPS["openai"] == [0, 1, 2, 3, 4], "OpenAI indices mismatch"
    assert FAMILY_GROUPS["google"] == [5, 6, 7, 8], "Google indices mismatch"
    assert FAMILY_GROUPS["anthropic"] == [9, 10, 11, 12, 13], "Anthropic indices mismatch"
    assert FAMILY_GROUPS["qwen"] == [14], "Qwen indices mismatch"
    assert FAMILY_GROUPS["zhipu"] == [15, 16], "Zhipu indices mismatch"
    assert FAMILY_GROUPS["moonshot"] == [17], "Moonshot indices mismatch"

    # Test numerical output shape and simplex property
    N = 20
    qwen_logits = np.random.randn(N, NUM_CLASSES)
    cpu_logits = np.random.randn(N, NUM_CLASSES)
    deb_logits = np.random.randn(N, NUM_CLASSES)
    bge_logits = np.random.randn(N, NUM_CLASSES)

    p_fused = hierarchical_tree_ensemble(qwen_logits, cpu_logits, deb_logits, bge_logits)
    assert p_fused.shape == (N, NUM_CLASSES), "Shape mismatch"
    assert np.all(np.isfinite(p_fused)), "Probabilities must be finite"
    assert np.allclose(p_fused.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1"

    # Test with None streams (graceful fallback)
    p_fallback = hierarchical_tree_ensemble(None, cpu_logits, deb_logits, None)
    assert p_fallback.shape == (N, NUM_CLASSES)
    assert np.allclose(p_fallback.sum(axis=1), 1.0, atol=1e-5)
    print("  [Pass] hierarchical_tree_ensemble verified.")

if __name__ == "__main__":
    test_sibling_difficulty_margins_focal_loss()
    test_nonlinear_stylometric_kernel_expander()
    test_dense_stylometric_classifier_kernel_expansion()
    test_prompt_sibling_paired_indices()
    test_same_prompt_sibling_repulsion()
    test_fast_batch_prefetcher_string_groups()
    test_global_ram_model_cache()
    test_hierarchical_tree_ensemble()
    print("\n>>> ALL RAM & SIBLING UPGRADE TESTS PASSED PERFECTLY! <<<")
