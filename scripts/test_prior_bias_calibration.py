import numpy as np
from sklearn.metrics import balanced_accuracy_score

K = 18
N = 1800  # 100 per class

# Test set has 100 samples per class (perfectly balanced ground truth)
y_true = np.repeat(np.arange(K), 100)

# Simulate prior bias from imbalanced training set
# class 0 has 1000 train samples (bias ~ +4.0), class 17 has 15 train samples (bias ~ -2.0)
train_counts = np.linspace(1000, 15, K)
prior_bias = np.log(train_counts / np.sum(train_counts))
prior_bias -= np.mean(prior_bias)  # zero-center

def make_biased_logits(signal_strength=3.5, noise_std=2.0):
    logits = np.random.randn(N, K) * noise_std
    for i in range(N):
        logits[i, y_true[i]] += signal_strength
    # Add training prior bias to simulate uncalibrated classifier
    logits += prior_bias[None, :] * 1.5
    return logits

np.random.seed(42)
deb_m = make_biased_logits(signal_strength=4.0, noise_std=2.0)
qwn_m = make_biased_logits(signal_strength=3.5, noise_std=2.0)
bge_m = make_biased_logits(signal_strength=3.0, noise_std=2.0)
svm_m = make_biased_logits(signal_strength=2.0, noise_std=2.5)

def compute_passage_ranks(margins):
    from scipy.stats import rankdata
    return np.apply_along_axis(rankdata, 1, margins)

def normalize_margins(margins):
    m_mean = margins.mean(axis=1, keepdims=True)
    m_std = np.maximum(margins.std(axis=1, keepdims=True), 1e-8)
    return (margins - m_mean) / m_std

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

# Quad blend WITHOUT Sinkhorn
total = (
    1.0 * (compute_passage_ranks(svm_m) + 0.3 * normalize_margins(svm_m))
    + 1.5 * (compute_passage_ranks(deb_m) + 0.3 * normalize_margins(deb_m))
    + 1.2 * (compute_passage_ranks(qwn_m) + 0.3 * normalize_margins(qwn_m))
    + 1.0 * (compute_passage_ranks(bge_m) + 0.3 * normalize_margins(bge_m))
)
raw_preds = np.argmax(total, axis=1)
raw_acc = balanced_accuracy_score(y_true, raw_preds)
print(f"Under Training Prior Bias:")
print(f"  Raw Quad Blend (uncalibrated) Balanced Acc: {raw_acc*100:.2f}%")
raw_counts = np.bincount(raw_preds, minlength=K)
print(f"  Raw Pred counts (min/max): {raw_counts.min()} / {raw_counts.max()}")

for T in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    P_bal = sinkhorn_knopp_balance(total, temperature=T)
    bal_preds = np.argmax(P_bal, axis=1)
    bal_acc = balanced_accuracy_score(y_true, bal_preds)
    bal_counts = np.bincount(bal_preds, minlength=K)
    print(f"  Sinkhorn (T={T:4.1f}) Balanced Acc: {bal_acc*100:.2f}% | counts (min/max): {bal_counts.min()} / {bal_counts.max()}")

