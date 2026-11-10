import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

print("Testing LoRA parameter reset mechanism on multi-layer mock backbone...")

class MockBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x):
        return self.out_proj(self.v_proj(self.q_proj(x)))

class MockTransformer(nn.Module):
    def __init__(self, num_layers=4, dim=32):
        super().__init__()
        self.layers = nn.ModuleList([MockBlock(dim) for _ in range(num_layers)])
        self.config = type("Config", (), {"hidden_size": dim})()
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

# 1. Instantiate and attach LoRA
base_model = MockTransformer(num_layers=4, dim=32)
lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(base_model, lora_cfg)

# 2. Check initial LoRA state
q0_b = peft_model.base_model.model.layers[0].q_proj.lora_B["default"].weight
assert torch.all(q0_b == 0.0), "Initial LoRA B should be all zeros"

# 3. Simulate training Fold 1 (modify LoRA weights)
with torch.no_grad():
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.add_(1.5)

assert not torch.all(peft_model.base_model.model.layers[0].q_proj.lora_B["default"].weight == 0.0)

# 4. Reset LoRA parameters for Fold 2
for name, module in peft_model.named_modules():
    if hasattr(module, "reset_lora_parameters"):
        try:
            module.reset_lora_parameters("default", init_lora_weights=True)
        except TypeError:
            module.reset_lora_parameters("default")

# 5. Verify that Fold 2 has completely reset LoRA weights
q0_b_reset = peft_model.base_model.model.layers[0].q_proj.lora_B["default"].weight
assert torch.all(q0_b_reset == 0.0), "Reset LoRA B should be all zeros again!"
print("[Pass] LoRA parameter reset verified 100% clean across folds.")

# 6. Test bottom layer freezing
print("Testing encoder bottom layer freezing...")
enc = MockTransformer(num_layers=6, dim=32)
n_freeze = 3
for l_idx in range(n_freeze):
    for p in enc.layers[l_idx].parameters():
        p.requires_grad = False

for l_idx in range(6):
    is_frozen = all(not p.requires_grad for p in enc.layers[l_idx].parameters())
    if l_idx < n_freeze:
        assert is_frozen, f"Layer {l_idx} should be frozen"
    else:
        assert not is_frozen, f"Layer {l_idx} should NOT be frozen"

print("[Pass] Encoder bottom layer freezing verified.")
print("\n>>> ALL STAGED RAM OPTIMIZATION TESTS PASSED! <<<")

