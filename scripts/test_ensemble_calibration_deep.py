import numpy as np
from sklearn.metrics import balanced_accuracy_score

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
K = len(LABELS)

FAMILY_MAP = {
    "gpt-5.5": "openai", "gpt-5.6-luna": "openai", "gpt-5.6-sol": "openai",
    "gpt-5.6-terra": "openai", "gpt-6-astra": "openai",
    "gemini-3.1-pro-preview": "google", "gemini-3.6-flash": "google",
    "gemini-3.7-flash": "google", "gemini-3.8-flash": "google",
    "claude-fable-5": "anthropic", "claude-opus-4-6": "anthropic",
    "claude-opus-4-8": "anthropic", "claude-opus-5": "anthropic", "claude-sonnet-5": "anthropic",
    "qwen3.8-max": "qwen",
    "glm-5.3": "zhipu", "glm-5.3-flash": "zhipu",
    "kimi-k3": "moonshot",
}
FAMILIES = sorted(set(FAMILY_MAP.values()))
FAMILY_TO_INDICES = {
    fam: [i for i, lab in enumerate(LABELS) if FAMILY_MAP[lab] == fam]
    for fam in FAMILIES
}

def compute_passage_ranks(margins):
    from scipy.stats import rankdata
    return np.apply_along_axis(rankdata, 1, margins)

def normalize_margins(margins):
    m_mean = margins.mean(axis=1, keepdims=True)
    m_std = np.maximum(margins.std(axis=1, keepdims=True), 1e-8)
    return (margins - m_mean) / m_std

def apply_hierarchical_family_gating(scores, alpha=0.35):
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.maximum(np.sum(np.exp(shifted), axis=1, keepdims=True), 1e-12)
    family_probs = np.zeros((scores.shape[0], len(FAMILIES)), dtype=np.float64)
    for f_idx, fam in enumerate(FAMILIES):
        family_probs[:, f_idx] = probs[:, FAMILY_TO_INDICES[fam]].sum(axis=1)

    gated = np.copy(scores)
    for f_idx, fam in enumerate(FAMILIES):
        fam_score = np.log(np.maximum(family_probs[:, f_idx : f_idx + 1], 1e-12))
        gated[:, FAMILY_TO_INDICES[fam]] += alpha * fam_score
    return gated

def sinkhorn_knopp_balance(scores, temperature=1.5, max_iters=50, tol=1e-3):
    N, K = scores.shape
    scaled_scores = scores / max(temperature, 1e-4)
    P = np.exp(scaled_scores - np.max(scaled_scores, axis=1, keepdims=True))
    target_col = np.full(K, N / K, dtype=np.float64)
    r = np.ones(N, dtype=np.float64)
    c = np.ones(K, dtype=np.float64)
    for _ in range(max_iters):
        c_prev = c.copy()
        c = target_col / (np.dot(r, P) + 1e-12)
        r = 1.0 / (np.dot(P, c) + 1e-12)
        if np.max(np.abs(c - c_prev)) < tol:
            break
    return r[:, None] * P * c[None, :]

# Simulate 1,000 test examples with balanced ground truth
N = 1000
np.random.seed(42)
y_true = np.random.randint(0, K, size=N)

# Generate realistic model logits: true class gets boost + noise
def make_logits(signal_strength=3.0, noise_std=2.0):
    logits = np.random.randn(N, K) * noise_std
    for i in range(N):
        logits[i, y_true[i]] += signal_strength
    return logits

svm_m = make_logits(signal_strength=2.0, noise_std=2.5)   # acc ~ 45%
deb_m = make_logits(signal_strength=4.0, noise_std=2.0)   # acc ~ 65%
qwn_m = make_logits(signal_strength=3.5, noise_std=2.0)   # acc ~ 60%
bge_m = make_logits(signal_strength=3.0, noise_std=2.0)   # acc ~ 55%

# Single model accuracies
for name, m in [("SVM", svm_m), ("DeBERTa", deb_m), ("Qwen", qwn_m), ("BGE", bge_m)]:
    acc = balanced_accuracy_score(y_true, np.argmax(m, axis=1))
    print(f"  {name} standalone bal acc: {acc*100:.2f}%")

# Quad blend WITHOUT Sinkhorn (raw argmax)
svm_clean = np.clip(svm_m, -50.0, 50.0)
deb_clean = np.clip(deb_m, -50.0, 50.0)
qwn_clean = np.clip(qwn_m, -50.0, 50.0)
bge_clean = np.clip(bge_m, -50.0, 50.0)

weights = (1.0, 1.5, 1.2, 1.0)
ws, wd, wq, wb = weights
total = (
    ws * (compute_passage_ranks(svm_clean) + 0.3 * normalize_margins(svm_clean))
    + wd * (compute_passage_ranks(deb_clean) + 0.3 * normalize_margins(deb_clean))
    + wq * (compute_passage_ranks(qwn_clean) + 0.3 * normalize_margins(qwn_clean))
    + wb * (compute_passage_ranks(bge_clean) + 0.3 * normalize_margins(bge_clean))
)
svm_ptp = np.ptp(svm_clean, axis=1, keepdims=True)
svm_norm = (svm_clean - svm_clean.min(axis=1, keepdims=True)) / np.maximum(svm_ptp, 1e-12)
total = total + 1e-6 * svm_norm

raw_blend_acc = balanced_accuracy_score(y_true, np.argmax(total, axis=1))
print(f"  Raw Quad Blend bal acc: {raw_blend_acc*100:.2f}%")

gated = apply_hierarchical_family_gating(total, alpha=0.35)
gated_acc = balanced_accuracy_score(y_true, np.argmax(gated, axis=1))
print(f"  + Family Gating bal acc: {gated_acc*100:.2f}%")

for T in [0.5, 1.0, 1.5, 2.0, 5.0, 10.0]:
    P_bal = sinkhorn_knopp_balance(gated, temperature=T)
    bal_acc = balanced_accuracy_score(y_true, np.argmax(P_bal, axis=1))
    print(f"  + Sinkhorn (T={T:4.1f}) bal acc: {bal_acc*100:.2f}%")

