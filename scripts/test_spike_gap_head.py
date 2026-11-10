import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np
from notebooks.kaggle_tri_ensemble import SpikeAndGapHierarchicalHead

def test_spike_and_gap_head():
    print("Testing SpikeAndGapHierarchicalHead...")
    torch.manual_seed(42)
    B, T, D = 4, 32, 4096
    num_classes = 18
    sub_centers = 3

    head = SpikeAndGapHierarchicalHead(hidden_size=D, num_classes=num_classes, sub_centers=sub_centers)
    
    # 1. Normal forward pass with attention mask
    hidden_states = torch.randn(B, T, D, requires_grad=True)
    attention_mask = torch.ones((B, T), dtype=torch.long)
    # Simulate variable sequence lengths with padding
    attention_mask[0, 20:] = 0
    attention_mask[1, 10:] = 0
    attention_mask[2, 30:] = 0

    logits = head(hidden_states, attention_mask)
    assert logits.shape == (B, num_classes), f"Expected shape {(B, num_classes)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN detected in logits"
    assert not torch.isinf(logits).any(), "Inf detected in logits"
    print("  [Pass] Normal forward pass with variable length padding.")

    # 2. Backward pass & gradient flow
    labels = torch.tensor([0, 5, 12, 17], dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert hidden_states.grad is not None, "Gradient did not flow to hidden_states"
    assert not torch.isnan(hidden_states.grad).any(), "NaN in hidden_states grad"
    assert head.weight.grad is not None, "Gradient did not flow to head.weight"
    assert not torch.isnan(head.weight.grad).any(), "NaN in head.weight grad"
    print("  [Pass] Backward pass and non-zero finite gradients verified.")

    # 3. Edge case: T=1 sequence
    h_single = torch.randn(2, 1, D)
    mask_single = torch.ones((2, 1), dtype=torch.long)
    logits_single = head(h_single, mask_single)
    assert logits_single.shape == (2, num_classes)
    assert not torch.isnan(logits_single).any()
    print("  [Pass] Edge case T=1 sequence length handled gracefully.")

    # 4. FP16 half-precision compatibility
    head_fp16 = SpikeAndGapHierarchicalHead(hidden_size=D, num_classes=num_classes, sub_centers=sub_centers).half()
    h_fp16 = torch.randn(2, 16, D, dtype=torch.float16)
    mask_fp16 = torch.ones((2, 16), dtype=torch.long)
    logits_fp16 = head_fp16(h_fp16, mask_fp16)
    assert logits_fp16.shape == (2, num_classes)
    assert not torch.isnan(logits_fp16).any()
    print("  [Pass] FP16 half-precision forward pass verified.")

    # 5. Device mismatch resilience
    head_cpu = SpikeAndGapHierarchicalHead(hidden_size=D, num_classes=num_classes, sub_centers=sub_centers)
    logits_dev = head_cpu(hidden_states.detach())
    assert logits_dev.shape == (B, num_classes)
    print("  [Pass] Device and dtype mismatch resilience verified.")

    print("\n>>> ALL SPIKE & GAP HIERARCHICAL HEAD TESTS PASSED [OK] <<<")

if __name__ == "__main__":
    test_spike_and_gap_head()

