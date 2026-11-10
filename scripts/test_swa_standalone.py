import torch
import torch.nn as nn
import numpy as np

# Test SWA tensor averaging on dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

m = DummyModel()
s1 = {k: v.clone() for k, v in m.state_dict().items()}
m.fc.weight.data += 1.0
s2 = {k: v.clone() for k, v in m.state_dict().items()}

states = [s1, s2]
swa_state = {}
for k in states[0]:
    if states[0][k].is_floating_point():
        swa_state[k] = torch.stack([s[k] for s in states]).mean(dim=0)
    else:
        swa_state[k] = states[-1][k]

diff = (swa_state["fc.weight"] - s1["fc.weight"]).abs().mean().item()
assert abs(diff - 0.5) < 1e-5, f"Expected 0.5 diff, got {diff}"
print(f"SWA dummy test passed! Diff={diff:.4f}")

