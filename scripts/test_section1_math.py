import numpy as np
import scipy.stats

# Test Calibrated Probability Blending + Smooth Family Gating + Relative Sinkhorn

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
FAMILY_MAP = {
    "gpt-5.5": "openai", "gpt-5.6-luna": "openai", "gpt-5.6-sol": "openai", "gpt-5.6-terra": "openai", "gpt-6-astra": "openai",
    "gemini-3.1-pro-preview": "google", "gemini-3.6-flash": "google", "gemini-3.7-flash": "google", "gemini-3.8-flash": "google",
    "claude-fable-5": "anthropic", "claude-opus-4-6": "anthropic", "claude-opus-4-8": "anthropic", "claude-opus-5": "anthropic", "claude-sonnet-5": "anthropic",
    "qwen3.8-max": "qwen",
    "glm-5.3": "zhipu", "glm-5.3-flash": "zhipu",
    "kimi-k3": "moonshot"
}
FAMILIES = sorted(set(FAMILY_MAP.values()))
FAMILY_TO_INDICES = {fam: [i for i, lab in enumerate(LABELS) if FAMILY_MAP[lab] == fam] for fam in FAMILIES}

def softmax(x, axis=1, temp=1.0):
    x_scaled = x / max(temp, 1e-4)
    e = np.exp(x_scaled - np.max(x_scaled, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def apply_hierarchical_family_gating_prob(probs, alpha=0.35):
    # Probs is (N, 18) summing to 1
    gated = probs.copy()
    for fam, indices in FAMILY_TO_INDICES.items():
        fam_mass = np.sum(probs[:, indices], axis=1, keepdims=True)
        # Smooth multiplicative scaling: fam_mass ** alpha
        gated[:, indices] *= (np.maximum(fam_mass, 1e-6) ** alpha)
    # Renormalize to valid probability simplex
    gated /= np.sum(gated, axis=1, keepdims=True)
    return gated

def sinkhorn_knopp_balance_prob(probs, max_iters=50, tol=1e-3):
    N, K = probs.shape
    P = np.copy(probs) + 1e-12
    target_col = np.full(K, N / K, dtype=np.float64)
    r = np.ones(N, dtype=np.float64)
    c = np.ones(K, dtype=np.float64)

    for it in range(max_iters):
        c_prev = c.copy()
        c = target_col / (np.dot(r, P) + 1e-12)
        r = 1.0 / (np.dot(P, c) + 1e-12)
        rel_diff = np.max(np.abs(c - c_prev)) / (np.max(np.abs(c)) + 1e-12)
        if rel_diff < tol:
            break

    P_bal = r[:, None] * P * c[None, :]
    # Renormalize rows
    P_bal /= np.sum(P_bal, axis=1, keepdims=True)
    return P_bal

# Test with 4000 test simulated rows
N = 4000
raw_logits = np.random.randn(N, 18) * 2.0
probs = softmax(raw_logits)
gated = apply_hierarchical_family_gating_prob(probs, alpha=0.35)
assert np.allclose(gated.sum(axis=1), 1.0)

balanced = sinkhorn_knopp_balance_prob(gated)
preds = np.argmax(balanced, axis=1)
counts = np.bincount(preds, minlength=18)
print("Sinkhorn Prob Balanced Test Class Distribution:")
print("Min count:", counts.min(), "Max count:", counts.max(), "Target:", N / 18)
assert counts.min() > 180, "Class starvation!"
print("Section 1.5 probability gating & Sinkhorn passed successfully!")

