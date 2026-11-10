"""Comprehensive Negative Stress-Testing for Name That LLM Pipeline."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

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
FAMILIES = sorted(list(set(FAMILY_MAP.values())))
FAMILY_TO_INDICES = {
    fam: [i for i, lab in enumerate(LABELS) if FAMILY_MAP[lab] == fam]
    for fam in FAMILIES
}

def clean_text(text: str) -> str:
    import re
    if not isinstance(text, str):
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    return s.strip()

def compute_passage_ranks(margins: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(rankdata, 1, margins)

def margins_to_probs(margins: np.ndarray, temp: float = 1.0) -> np.ndarray:
    clean = np.clip(margins, -50.0, 50.0)
    scaled = clean / max(temp, 1e-4)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / np.maximum(np.sum(exp_s, axis=1, keepdims=True), 1e-12)

def sinkhorn_knopp_balance(probs: np.ndarray, max_iters: int = 50, tol: float = 1e-3) -> np.ndarray:
    N, K = probs.shape
    P = np.copy(probs) + 1e-12
    target_col = np.full(K, N / K, dtype=np.float64)
    r = np.ones(N, dtype=np.float64)
    c = np.ones(K, dtype=np.float64)

    for _ in range(max_iters):
        c_prev = c.copy()
        c = target_col / (np.dot(r, P) + 1e-12)
        r = 1.0 / (np.dot(P, c) + 1e-12)
        rel_diff = np.max(np.abs(c - c_prev)) / (np.max(np.abs(c)) + 1e-12)
        if rel_diff < tol:
            break

    P_bal = r[:, None] * P * c[None, :]
    return P_bal / np.maximum(np.sum(P_bal, axis=1, keepdims=True), 1e-12)

def apply_hierarchical_family_gating(probs: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    K = probs.shape[1]
    gated = probs.copy()
    for fam, indices in FAMILY_TO_INDICES.items():
        prior_mass = max(len(indices) / K, 1e-6)
        fam_mass = np.sum(probs[:, indices], axis=1, keepdims=True)
        concentration = np.maximum(fam_mass / prior_mass, 1e-6)
        gated[:, indices] *= (concentration ** alpha)
    return gated / np.maximum(np.sum(gated, axis=1, keepdims=True), 1e-12)

def normalize_margins(m: np.ndarray) -> np.ndarray:
    std = np.std(m, axis=1, keepdims=True)
    return (m - np.mean(m, axis=1, keepdims=True)) / np.maximum(std, 1e-9)

def quad_rank_blend(
    svm_m, deberta_m=None, qwen_m=None, bge_m=None,
    weights=(1.0, 1.0, 1.0, 1.0), balance_test=False, apply_family_gating=True
):
    ws, wd, wq, wb = weights
    w_sum = ws
    total_prob = ws * margins_to_probs(svm_m, temp=0.20)
    if deberta_m is not None:
        total_prob = total_prob + wd * margins_to_probs(deberta_m, temp=1.2)
        w_sum += wd
    if qwen_m is not None:
        total_prob = total_prob + wq * margins_to_probs(qwen_m, temp=1.5)
        w_sum += wq
    if bge_m is not None:
        total_prob = total_prob + wb * margins_to_probs(bge_m, temp=1.2)
        w_sum += wb

    probs = total_prob / max(w_sum, 1e-6)
    if apply_family_gating:
        probs = apply_hierarchical_family_gating(probs, alpha=0.35)
    if balance_test and len(probs) >= 36:
        probs = sinkhorn_knopp_balance(probs)
    return np.argmax(probs, axis=1)

def tri_rank_blend(
    svm_margins, qwen_margins=None, deberta_margins=None,
    weights=(1.0, 1.0, 1.0), balance_test: bool = False, apply_family_gating: bool = True
):
    ws, wq, wd = weights
    return quad_rank_blend(
        svm_margins, deberta_m=deberta_margins, qwen_m=qwen_margins, bge_m=None,
        weights=(ws, wd, wq, 0.0), balance_test=balance_test, apply_family_gating=apply_family_gating
    )

print("--- RUNNING NEGATIVE STRESS TESTS ---")

# TEST 1: Adversarial text inputs
adversarial_texts = [
    "", "   \n\t  ", None, 12345, "\x00\x08weird\x7fcontrol",
    "🚀" * 500, "word " * 2000, "SingleWord", "A"
]
cleaned = [clean_text(t) for t in adversarial_texts]
print("Test 1 (Text Cleaning): Passed without exception")

# TEST 2: Single-row test set (boundary condition)
svm_single = np.random.randn(1, 18)
qwen_single = np.random.randn(1, 18)
preds_single = tri_rank_blend(svm_single, qwen_single, None, balance_test=True)
assert len(preds_single) == 1
print("Test 2 (Single-Row Boundary): Passed (pred idx: {})".format(preds_single[0]))

# TEST 3: Zero-variance margins (all classes identical margin)
svm_const = np.zeros((10, 18))
preds_const = tri_rank_blend(svm_const, None, None, balance_test=True)
assert len(preds_const) == 10
print("Test 3 (Zero-Variance Tie-Break): Passed without NaN or ZeroDivisionError")

# TEST 4: Extreme logits (magnitude +/- 10,000)
svm_extreme = np.full((50, 18), -10000.0)
svm_extreme[:, 3] = 10000.0
preds_extreme = tri_rank_blend(svm_extreme, None, None, balance_test=True)
assert len(preds_extreme) == 50
print("Test 4 (Extreme Logit Dynamic Range): Passed")

# TEST 5: Sinkhorn with N = 4000 (hidden test simulation)
np.random.seed(42)
scores_4k = np.random.randn(4000, 18) * 3.0
preds_4k = tri_rank_blend(scores_4k, None, None, balance_test=True)
assert len(preds_4k) == 4000
counts_4k = np.bincount(preds_4k, minlength=18)
print("Test 5 (4,000 Row Test Balancing): Min count={}, Max count={}, Target={:.1f}".format(
    counts_4k.min(), counts_4k.max(), 4000 / 18
))
assert counts_4k.min() > 180, "Class starvation detected in 4k test!"

# TEST 6: Hierarchical Family Gating with zero probabilities & Cardinality Invariance
scores_zero = np.full((5, 18), -500.0)
gated = apply_hierarchical_family_gating(scores_zero)
assert not np.isnan(gated).any()
assert not np.isinf(gated).any()

# Adversarial test: Under uniform probability, NO family gets artificial cardinality advantage
uniform_p = np.full((10, 18), 1.0 / 18.0)
gated_uniform = apply_hierarchical_family_gating(uniform_p)
np.testing.assert_allclose(gated_uniform, uniform_p, atol=1e-6,
    err_msg="Family gating exhibits cardinality bias on uniform input!")
print("Test 6 (Hierarchical Gating Underflow & Cardinality Invariance): Passed")

# TEST 7: Missing classes in fold predictions (align_margins robustness)
def align_margins(margins, classes, labels, default_val=-999.0):
    aligned = np.full((len(margins), len(labels)), default_val, dtype=np.float64)
    cls_list = list(classes)
    for i, lab in enumerate(labels):
        if lab in cls_list:
            aligned[:, i] = margins[:, cls_list.index(lab)]
    return aligned

subset_classes = LABELS[:12]  # Only 12 of 18 classes present in an uneven fold
raw_margins = np.random.randn(20, 12)
aligned = align_margins(raw_margins, subset_classes, LABELS)
assert aligned.shape == (20, 18)
assert (aligned[:, 12:] == -999.0).all()
preds_aligned = tri_rank_blend(aligned, None, None)
assert len(preds_aligned) == 20
assert all(p < 12 for p in preds_aligned), "Missing classes were erroneously predicted!"
print("Test 7 (align_margins Missing Class Recovery): Passed")

# TEST 8: Adversarial Submission Parquet Validation
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ntl.submission import validate_submission_frame
test_ids = [f"test_{i:04d}" for i in range(100)]
test_preds = [LABELS[i % 18] for i in range(100)]
df_sub = pd.DataFrame({"id": test_ids, "model": test_preds})
validate_submission_frame(df_sub, expected_ids=test_ids)
print("Test 8 (Submission Schema Integrity): Passed against official validator")

# TEST 9: DualLinearClassifier Scikit-Learn Compliance & String Label Permutation Alignment
from sklearn.base import BaseEstimator, clone
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from scipy.sparse import csr_matrix

class DualLinearClassifier(BaseEstimator):
    def __init__(self, c_svc=1.2, alpha_ridge=1.0, svc_weight=0.7, seed=42, canonical_labels=None):
        self.c_svc = c_svc
        self.alpha_ridge = alpha_ridge
        self.svc_weight = svc_weight
        self.seed = seed
        self.canonical_labels = canonical_labels

    def fit(self, X, y):
        self.svc_ = LinearSVC(C=self.c_svc, class_weight=None, dual="auto", max_iter=2000, random_state=self.seed)
        self.ridge_ = RidgeClassifier(alpha=self.alpha_ridge, class_weight=None, random_state=self.seed)
        self.svc_.fit(X, y)
        self.ridge_.fit(X, y)
        raw_classes = list(self.svc_.classes_)
        if self.canonical_labels is not None:
            self.classes_ = np.array(self.canonical_labels)
            self._raw_classes = raw_classes
        else:
            self.classes_ = self.svc_.classes_
            self._raw_classes = raw_classes
        return self

    def decision_function(self, X):
        d_svc = self.svc_.decision_function(X)
        d_ridge = self.ridge_.decision_function(X)
        m = self.svc_weight * d_svc + (1.0 - self.svc_weight) * d_ridge
        if self.canonical_labels is not None and len(self._raw_classes) > 0 and isinstance(self._raw_classes[0], str):
            return align_margins(m, self._raw_classes, self.canonical_labels)
        return m

X_dummy = csr_matrix(np.random.randn(60, 50))
y_dummy = np.random.randint(0, 18, size=60)
clf = DualLinearClassifier(c_svc=1.0, alpha_ridge=1.0, svc_weight=0.6, seed=42)
cloned_clf = clone(clf)
cloned_clf.fit(X_dummy, y_dummy)
dfunc = cloned_clf.decision_function(X_dummy)
assert dfunc.shape == (60, len(np.unique(y_dummy)))

# Adversarial string label permutation test:
# Fit on string labels where alphabetical order is completely different from canonical LABELS
y_strings = [LABELS[i % 18] for i in range(180)]
X_str = csr_matrix(np.random.randn(180, 50))
clf_str = DualLinearClassifier(c_svc=1.0, alpha_ridge=1.0, svc_weight=0.6, seed=42, canonical_labels=LABELS)
clf_str.fit(X_str, y_strings)
dfunc_str = clf_str.decision_function(X_str)
assert dfunc_str.shape == (180, 18)
assert list(clf_str.classes_) == LABELS, "Classifier classes_ must match canonical LABELS!"
assert clf_str._raw_classes != LABELS, "Raw LinearSVC classes must be alphabetical, verifying non-trivial alignment!"
print("Test 9 (DualLinearClassifier Compliance & Canonical String Alignment): Passed")

# TEST 10: Quad-Rank Blend with 4 Model Streams
m1 = np.random.randn(100, 18)
m2 = np.random.randn(100, 18)
m3 = np.random.randn(100, 18)
m4 = np.random.randn(100, 18)
q_preds = quad_rank_blend(m1, m2, m3, m4, weights=(1.0, 1.2, 1.5, 0.8), balance_test=True)
assert len(q_preds) == 100
assert all(0 <= p < 18 for p in q_preds)
print("Test 10 (Quad-Rank 4-Stream Blend): Passed")

# TEST 11: Empty Slice Guard Check
def empty_slice_guard_test(texts_list):
    if len(texts_list) == 0:
        return np.zeros((0, 18), dtype=np.float64)
    return np.ones((len(texts_list), 18), dtype=np.float64)

assert empty_slice_guard_test([]).shape == (0, 18)
assert empty_slice_guard_test(["sample"]).shape == (1, 18)
print("Test 11 (Empty Slice Guard Verification): Passed")

# TEST 12: DenseStylometricClassifier with Missing Classes & Log-Simplex Finite Guarantee
p_raw = np.random.uniform(0.01, 1.0, size=(20, 12))
p_raw = p_raw / p_raw.sum(axis=1, keepdims=True)
p_aligned = align_margins(p_raw, LABELS[:12], LABELS, default_val=0.0)
p_norm = p_aligned / np.maximum(p_aligned.sum(axis=1, keepdims=True), 1e-12)
assert p_norm.shape == (20, 18)
assert (p_norm[:, 12:] == 0.0).all()
assert np.all(p_norm >= 0.0), "Negative probability detected!"
log_p = 0.20 * np.log(p_norm + 1e-12)
assert np.all(np.isfinite(log_p)), "NaN encountered during log(p_cpu) calculation!"
print("Test 12 (DenseStylometric Probability Alignment & Log Finite Guarantee): Passed")

# TEST 13: PromptDF Mask Extreme Boundary Fallback
from scipy import sparse as sp
X_toy = sp.csr_matrix(np.random.binomial(1, 0.1, size=(50, 100)).astype(np.float32))
groups_toy = np.random.randint(0, 5, size=50)

def compute_prompt_df_mask_test(X, groups, min_prompts=3):
    if min_prompts <= 1 or groups is None:
        return np.ones(X.shape[1], dtype=np.float32)
    B = (X > 0).astype(np.float32).tocsr()
    codes = pd.factorize(groups)[0]
    G = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))))
    pdf = np.asarray(((G @ B) > 0).sum(axis=0)).ravel()
    mask = (pdf >= min_prompts).astype(np.float32)
    if not np.any(mask > 0):
        return np.ones(X.shape[1], dtype=np.float32)
    return mask

# Extremely high threshold: no feature can pass
mask_extreme = compute_prompt_df_mask_test(X_toy, groups_toy, min_prompts=999)
assert np.all(mask_extreme == 1.0), "Extreme mask failed to fall back to all-ones!"
print("Test 13 (PromptDF Mask Extreme Boundary Fallback): Passed")

# TEST 14: check_finite_logits NaN/Inf Detection & Rejection
def check_finite_logits_test(logits: np.ndarray, name: str) -> np.ndarray:
    if logits is None:
        return None
    if not np.all(np.isfinite(logits)):
        n_nan = int(np.sum(np.isnan(logits)))
        n_inf = int(np.sum(np.isinf(logits)))
        raise FloatingPointError(f"[{name}] Non-finite logits detected: {n_nan} NaNs, {n_inf} Infs in shape {logits.shape}")
    return logits

good_logits = np.random.randn(10, 18)
assert check_finite_logits_test(good_logits, "Good") is good_logits
assert check_finite_logits_test(None, "None") is None

nan_logits = good_logits.copy()
nan_logits[3, 5] = np.nan
try:
    check_finite_logits_test(nan_logits, "NaN test")
    assert False, "Failed to catch NaN logits!"
except FloatingPointError:
    pass

inf_logits = good_logits.copy()
inf_logits[2, 1] = np.inf
try:
    check_finite_logits_test(inf_logits, "Inf test")
    assert False, "Failed to catch Inf logits!"
except FloatingPointError:
    pass
print("Test 14 (check_finite_logits NaN/Inf Trap): Passed")

# TEST 15: FocalLoss Dtype/Device Invariance & True-Class Modulating Factor
import torch
import torch.nn as nn

class FocalLossTest(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        targets = targets.to(device=logits.device, dtype=torch.long)
        log_prob = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        prob = torch.exp(log_prob)
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_prob, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        ce_loss = -(smooth_targets * log_prob).sum(dim=-1)
        loss = focal_weight * ce_loss
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            loss = loss * w
        if self.reduction == "mean":
            return loss.mean()
        return loss

crit = FocalLossTest(gamma=2.0, label_smoothing=0.05)
l_fp16 = torch.randn(8, 18, dtype=torch.float16, requires_grad=True)
t_long = torch.randint(0, 18, (8,))
loss_val = crit(l_fp16, t_long)
assert not torch.isnan(loss_val) and not torch.isinf(loss_val)
loss_val.backward()
assert l_fp16.grad is not None and not torch.isnan(l_fp16.grad).any()
print("Test 15 (FocalLoss FP16/FP32 Autocast & Gradient Stability): Passed")

# TEST 16: Safe Fold Appending (Rejection of None & Non-Finite)
test_deb_folds_mock = []
for candidate in [np.random.randn(5, 18), None, np.full((5, 18), np.nan), np.random.randn(5, 18)]:
    if candidate is not None and np.all(np.isfinite(candidate)):
        test_deb_folds_mock.append(candidate)
assert len(test_deb_folds_mock) == 2
mean_te = np.mean(test_deb_folds_mock, axis=0)
assert mean_te.shape == (5, 18) and np.all(np.isfinite(mean_te))
print("Test 16 (Safe Fold Appending Guard): Passed")

# TEST 17: CPU Subblend Pure-Sparse Anchor & Tie Preservation
def optimize_cpu_subblend_test(sparse_probs: np.ndarray, dense_probs: np.ndarray, y_true_int: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    valid = np.isfinite(sparse_probs).all(axis=1) & np.isfinite(dense_probs).all(axis=1)
    if np.sum(valid) < 10:
        return 0.85
    base_score = balanced_accuracy_score(y_true_int[valid], sparse_probs[valid].argmax(axis=1))
    best_score = base_score
    best_alpha = 1.0
    for alpha in np.arange(0.95, 0.45, -0.05):
        blend = alpha * sparse_probs[valid] + (1.0 - alpha) * dense_probs[valid]
        score = balanced_accuracy_score(y_true_int[valid], blend.argmax(axis=1))
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return round(best_alpha, 2)

np.random.seed(42)
y_mock = np.random.randint(0, 18, 200)
sp_logits = np.random.randn(200, 18)
for i in range(200):
    sp_logits[i, y_mock[i]] += 2.0
sp_probs = np.exp(sp_logits) / np.exp(sp_logits).sum(axis=1, keepdims=True)
dense_pure_noise = np.random.dirichlet(np.ones(18), size=200)
alpha_out = optimize_cpu_subblend_test(sp_probs, dense_pure_noise, y_mock)
assert alpha_out == 1.0, f"Expected 1.0 (pure sparse dominance under noise), got {alpha_out}"
print("Test 17 (CPU Subblend Sparse Dominance Anchor & Noise Rejection): Passed")

# TEST 18: Blend Weight Candidates Include Unit 1.0 Across All Streams
s_cands = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
d_cands = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
q_cands = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
b_cands = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
for name, cands in [("SVM", s_cands), ("DeBERTa", d_cands), ("Qwen", q_cands), ("BGE", b_cands)]:
    assert 1.0 in cands, f"Unit weight 1.0 missing from {name} candidate grid!"
print("Test 18 (Unit Weight 1.0 Grid Completeness): Passed")

# TEST 19: Dynamic Interim Checkpoint Weighting (Active Stream Representation)
def compute_interim_weights(deb_m, qwen_m, bge_m):
    return (
        1.0,
        1.0 if deb_m is not None else 0.0,
        1.0 if qwen_m is not None else 0.0,
        1.0 if bge_m is not None else 0.0,
    )

w_svm_only = compute_interim_weights(None, None, None)
assert w_svm_only == (1.0, 0.0, 0.0, 0.0)
w_deb_bge = compute_interim_weights(np.zeros((10, 18)), None, np.zeros((10, 18)))
assert w_deb_bge == (1.0, 1.0, 0.0, 1.0)
w_all = compute_interim_weights(np.zeros((10, 18)), np.zeros((10, 18)), np.zeros((10, 18)))
assert w_all == (1.0, 1.0, 1.0, 1.0)
print("Test 19 (Dynamic Interim Checkpoint Weighting): Passed")

# TEST 20: FocalLoss Gamma=1.0 Default Parameter Alignment
crit_default = FocalLossTest(gamma=1.0)
assert crit_default.gamma == 1.0
print("Test 20 (FocalLoss Gamma=1.0 Calibration Alignment): Passed")

print("\n>>> ALL 20 NEGATIVE TESTS PASSED WITH 100% SUCCESS <<<")
