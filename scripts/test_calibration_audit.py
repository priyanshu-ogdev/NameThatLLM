import numpy as np

# Simulate total rank scores across 4 models
# Rank of class k in [1, 18], sum of 4 models is in [4, 72]
np.random.seed(42)
N = 100
K = 18

# Model 1, 2, 3, 4 ranks
ranks = np.random.randint(1, 19, size=(4, N, K))
weights = [1.0, 1.0, 1.0, 1.0]
total = sum(w * r for w, r in zip(weights, ranks)) # range ~ 10 to 65

def sinkhorn_knopp_balance(scores, temperature=1.5, max_iters=50, tol=1e-3):
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

for T in [1.5, 5.0, 10.0, 15.0, 20.0]:
    scaled = total / T
    P_init = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    P_init /= P_init.sum(axis=1, keepdims=True)
    entropy = -np.mean(np.sum(P_init * np.log(np.maximum(P_init, 1e-12)), axis=1))
    top1_conf = np.mean(np.max(P_init, axis=1))
    
    P_bal = sinkhorn_knopp_balance(total, temperature=T)
    preds = np.argmax(P_bal, axis=1)
    counts = np.bincount(preds, minlength=K)
    print(f"T={T:4.1f} | Top1 Conf: {top1_conf*100:5.1f}% | Entropy: {entropy:4.2f} | Pred Min/Max/Std: {counts.min()}/{counts.max()}/{counts.std():.1f}")

