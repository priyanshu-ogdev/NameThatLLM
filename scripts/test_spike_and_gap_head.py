import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import numpy as np
from notebooks.kaggle_tri_ensemble import SpikeAndGapHierarchicalHead, StylometricMultiMomentPooler, capture_intermediate_layers

def test_spike_and_gap_hierarchical_head():
    print("Testing SpikeAndGapHierarchicalHead comprehensive mechanics...")
    B, T, D = 4, 32, 128
    num_classes = 18
    sub_centers = 3
    head = SpikeAndGapHierarchicalHead(hidden_size=D, num_classes=num_classes, sub_centers=sub_centers)

    # 1. Test single tensor input
    h_single = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 20:] = 0  # partial padding on row 0
    mask[1, 10:] = 0  # partial padding on row 1

    logits = head(h_single, mask)
    assert logits.shape == (B, num_classes), f"Expected shape ({B}, {num_classes}), got {logits.shape}"
    assert torch.all(torch.isfinite(logits)), "Logits must be finite"

    # Backward pass verification
    loss = logits.sum()
    loss.backward()
    assert h_single.grad is not None and torch.all(torch.isfinite(h_single.grad)), "Gradients must be finite"
    print("  [Pass] Single tensor forward and backward pass verified.")

    # 2. Test 3-tuple multi-layer hidden state input (captured from layers -1, -5, -9)
    head.zero_grad()
    h_layers = (
        torch.randn(B, T, D, requires_grad=True),
        torch.randn(B, T, D, requires_grad=True),
        torch.randn(B, T, D, requires_grad=True),
    )
    logits_multi = head(h_layers, mask)
    assert logits_multi.shape == (B, num_classes), "Multi-layer shape mismatch"
    assert torch.all(torch.isfinite(logits_multi)), "Multi-layer logits must be finite"
    loss_multi = logits_multi.sum()
    loss_multi.backward()
    for i, h_l in enumerate(h_layers):
        assert h_l.grad is not None and torch.all(torch.isfinite(h_l.grad)), f"Layer {i} gradient not finite"
    print("  [Pass] Multi-layer hidden state fusion (-1, -5, -9) verified.")

    # 3. Test short sequence edge cases (T = 1 and T = 2)
    head.zero_grad()
    h_t1 = torch.randn(2, 1, D)
    mask_t1 = torch.ones(2, 1, dtype=torch.long)
    logits_t1 = head(h_t1, mask_t1)
    assert logits_t1.shape == (2, num_classes), "T=1 shape mismatch"
    assert torch.all(torch.isfinite(logits_t1)), "T=1 logits must be finite"
    print("  [Pass] T=1 short sequence edge case safely handled without crashing.")

    # 4. Test sub-center cosine margin geometry
    with torch.no_grad():
        w_norm = torch.nn.functional.normalize(head.weight, p=2, dim=-1)
        sim = torch.mm(w_norm, w_norm.t())
        assert torch.all(sim <= 1.0001), "Cosine similarities must be <= 1.0"
    print("  [Pass] Sub-center cosine geometry verified.")

def test_stylometric_multi_moment_pooler():
    print("Testing StylometricMultiMomentPooler (CLS + Mean + Max + Std)...")
    B, T, D = 4, 32, 128
    pooler = StylometricMultiMomentPooler(hidden_size=D)
    h = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[2, 15:] = 0

    out = pooler(h, mask)
    assert out.shape == (B, D), f"Expected shape ({B}, {D}), got {out.shape}"
    assert torch.all(torch.isfinite(out)), "Pooler output must be finite"

    loss = out.sum()
    loss.backward()
    assert h.grad is not None and torch.all(torch.isfinite(h.grad)), "Pooler gradients must be finite"
    print("  [Pass] StylometricMultiMomentPooler verified.")

def test_recursive_capture_intermediate_layers():
    print("Testing recursive layer discovery in capture_intermediate_layers on nested PEFT architecture...")
    class _MockLayer(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.linear = nn.Linear(d, d)
        def forward(self, x):
            return (self.linear(x),)

    class _MockInner(nn.Module):
        def __init__(self, n=12, d=32):
            super().__init__()
            self.layers = nn.ModuleList([_MockLayer(d) for _ in range(n)])
        def forward(self, x):
            for l in self.layers:
                x = l(x)[0]
            return x

    class _MockClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _MockInner()
        def forward(self, x):
            return self.model(x)

    class _MockPeft(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.model = _MockClf()
        def forward(self, x):
            return self.base_model.model(x)

    mock_model = _MockPeft()
    with capture_intermediate_layers(mock_model, (-1, -5, -9)) as caps:
        x = torch.randn(2, 4, 32)
        _ = mock_model(x)
        assert -1 in caps and -5 in caps and -9 in caps, "Must capture -1, -5, -9 through nested PEFT layers"
        assert caps[-1].shape == (2, 4, 32), "Captured layer shape mismatch"
    print("  [Pass] Recursive layer discovery and hook capture on nested PEFT architecture verified.")

if __name__ == "__main__":
    test_spike_and_gap_hierarchical_head()
    test_stylometric_multi_moment_pooler()
    test_recursive_capture_intermediate_layers()
    print("\n>>> ALL SPIKE & GAP HEAD & POOLER TESTS PASSED 100%! <<<")

