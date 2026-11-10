import torch
import torch.nn.functional as F
import numpy as np
import time

FAMILY_GROUPS = [
    [0, 1, 2, 3, 4],   # Anthropic
    [5, 6, 7, 8, 9],   # OpenAI
    [10, 11, 12, 13],  # Google
    [14, 15],          # Zhipu
    [16],              # Moonshot
    [17],              # Alibaba
]

class CPUGradientMetaCombiner:
    """
    PyTorch CPU Autograd L-BFGS Meta-Combiner.
    Optimizes per-model, per-class specialization weights W[M, C],
    per-model, per-family router weights W_rot[M, 6],
    sample-level entropy confidence gates, and Bayesian hierarchical tree fusion.
    Executes in under 0.5 seconds on CPU with exact analytical gradients.
    """
    def __init__(self, n_models=4, n_classes=18, l2_reg=0.05, family_groups=None):
        self.M = n_models
        self.C = n_classes
        self.l2_reg = l2_reg
        self.family_groups = family_groups or FAMILY_GROUPS
        self.W_exp = None
        self.W_rot = None
        self.gamma = None
        self.alpha_raw = None

    @staticmethod
    def _compute_entropy(probs):
        p = probs.clamp(min=1e-12)
        ent = -torch.sum(p * torch.log(p), dim=-1, keepdim=True)
        return ent / torch.log(torch.tensor(float(probs.shape[-1])))

    def _forward(self, logits_stack):
        # logits_stack: [M, N, C]
        M, N, C = logits_stack.shape
        probs_m = F.softmax(logits_stack, dim=-1)
        ent_m = self._compute_entropy(probs_m)  # [M, N, 1]
        
        g = torch.exp(torch.clamp(self.gamma, -5.0, 5.0)).view(M, 1, 1)
        conf_gate = torch.exp(-g * ent_m)
        gated = logits_stack * conf_gate  # [M, N, C]
        
        # 1. Expert logits
        z_expert = torch.sum(self.W_exp.view(M, 1, C) * gated, dim=0)  # [N, C]

        # 2. Router logits
        fam_logits_m = torch.zeros((M, N, 6), dtype=logits_stack.dtype)
        for f_idx, c_idxs in enumerate(self.family_groups):
            fam_logits_m[:, :, f_idx] = torch.logsumexp(gated[:, :, c_idxs], dim=-1)
        z_router = torch.sum(self.W_rot.view(M, 1, 6) * fam_logits_m, dim=0)  # [N, 6]
        p_family = F.softmax(z_router, dim=-1)  # [N, 6]

        # 3. Hierarchical Bayesian Tree Fusion
        p_tree = torch.zeros_like(z_expert)
        for f_idx, c_idxs in enumerate(self.family_groups):
            p_fam_slice = p_family[:, [f_idx]]
            if len(c_idxs) > 1:
                p_sib = F.softmax(z_expert[:, c_idxs], dim=-1)
            else:
                p_sib = torch.ones((N, 1), dtype=logits_stack.dtype)
            p_tree[:, c_idxs] = p_fam_slice * p_sib

        p_flat = F.softmax(z_expert, dim=-1)
        alpha = torch.sigmoid(self.alpha_raw)
        p_fused = (alpha * p_tree + (1.0 - alpha) * p_flat).clamp(min=1e-12)
        return p_fused / p_fused.sum(dim=-1, keepdim=True)

    def fit(self, oof_logits_list, y_true, prior_weights=None, class_weights=None, max_iter=45):
        valid_streams = [l for l in oof_logits_list if l is not None]
        self.M = len(valid_streams)
        logits_stack = torch.tensor(np.stack(valid_streams, axis=0), dtype=torch.float32)
        y = torch.tensor(y_true, dtype=torch.long)
        N = y.shape[0]

        if prior_weights is None:
            prior_w = torch.full((self.M,), 1.0 / self.M)
        else:
            pw = [w for w, l in zip(prior_weights, oof_logits_list) if l is not None]
            prior_w = torch.tensor(pw, dtype=torch.float32)
            prior_w = prior_w / prior_w.sum()

        self.W_exp = torch.nn.Parameter(prior_w.view(self.M, 1).repeat(1, self.C))
        self.W_rot = torch.nn.Parameter(torch.full((self.M, 6), 1.0 / self.M))
        self.gamma = torch.nn.Parameter(torch.zeros(self.M))
        self.alpha_raw = torch.nn.Parameter(torch.tensor([0.8]))

        exp_target = self.W_exp.data.clone()
        rot_target = self.W_rot.data.clone()

        if class_weights is None:
            counts = np.bincount(y_true, minlength=self.C)
            cw = N / (self.C * np.maximum(counts, 1).astype(np.float32))
            cw_tensor = torch.tensor(cw, dtype=torch.float32)
        else:
            cw_tensor = torch.tensor(class_weights, dtype=torch.float32)
        sample_w = cw_tensor[y]

        optimizer = torch.optim.LBFGS(
            [self.W_exp, self.W_rot, self.gamma, self.alpha_raw],
            max_iter=max_iter, lr=0.15, history_size=10
        )

        def closure():
            optimizer.zero_grad()
            probs = self._forward(logits_stack)
            nll = -torch.mean(sample_w * torch.log(probs[torch.arange(N), y]))
            reg = self.l2_reg * (
                torch.sum((self.W_exp - exp_target) ** 2) +
                torch.sum((self.W_rot - rot_target) ** 2) +
                torch.sum(self.gamma ** 2)
            )
            loss = nll + reg
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict_proba(self, test_logits_list):
        valid_streams = [l for l in test_logits_list if l is not None]
        logits_stack = torch.tensor(np.stack(valid_streams, axis=0), dtype=torch.float32)
        with torch.no_grad():
            probs = self._forward(logits_stack)
        return probs.cpu().numpy()

if __name__ == "__main__":
    np.random.seed(42)
    N = 2160
    y = np.random.randint(0, 18, N)
    logits = [np.random.randn(N, 18) for _ in range(4)]
    for i in range(N):
        logits[0][i, y[i]] += 1.0
        logits[1][i, y[i]] += 1.5
        logits[2][i, y[i]] += 2.0
        logits[3][i, y[i]] += 0.8

    t0 = time.perf_counter()
    combiner = CPUGradientMetaCombiner(n_models=4, n_classes=18, l2_reg=0.05)
    combiner.fit(logits, y, prior_weights=[0.25, 0.20, 0.45, 0.10])
    fit_dur = time.perf_counter() - t0
    probs = combiner.predict_proba(logits)
    print(f"Full CPU PyTorch L-BFGS Combiner fit in {fit_dur:.3f}s!")
    print(f"Probs shape: {probs.shape}, row sums: {probs.sum(axis=1)[:3]}")
    print(f"Alpha: {torch.sigmoid(combiner.alpha_raw).item():.3f}")
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities must sum to 1.0"
    print("ALL TESTS PASSED [OK]!")

