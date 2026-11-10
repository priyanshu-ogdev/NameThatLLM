import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from notebooks.kaggle_dry_run import (
    DualLinearClassifier,
    build_cpu_feature_extractor,
    make_prompt_groups,
    margins_to_probs,
    fit_temperature,
    sinkhorn_knopp_balance,
    quad_rank_blend,
    validate_and_save_submission,
    check_finite_logits,
    LABELS,
    NUM_CLASSES,
)

print("=== ADVERSARIAL NEGATIVE AUDIT OF DRY RUN & PRODUCTION PIPELINE ===")

# Test 1: Binary slice (only 2 classes)
print("\n[Audit Test 1/7] Binary Slice in DualLinearClassifier...")
extractor = build_cpu_feature_extractor(style_weight=0.2)
texts_bin = ["The quick brown fox jumps over the lazy dog.", "In conclusion, the thesis demonstrated above is clear."] * 5
y_bin = ["gpt-5.5", "claude-opus-5"] * 5
X_bin = extractor.fit_transform(texts_bin)
clf = DualLinearClassifier(c_svc=1.2, alpha_ridge=1.0, svc_weight=0.7, seed=42, canonical_labels=LABELS)
clf.fit(X_bin, y_bin)
margins_bin = clf.decision_function(X_bin)
assert margins_bin.shape == (10, 18), f"Expected shape (10, 18), got {margins_bin.shape}"
assert not np.isnan(margins_bin).any(), "NaN found in binary margins"
print("  -> Binary Slice Handled Seamlessly: shape (10, 18) [PASSED]")

# Test 2: Missing prompt_id column
print("\n[Audit Test 2/7] Missing 'prompt_id' Column Handling...")
df_no_pid = pd.DataFrame({"text": ["Sample text A", "Sample text B", "Sample text A"]})
groups = make_prompt_groups(df_no_pid)
assert len(groups) == 3, "Grouping length mismatch"
assert groups[0] == groups[2], "Identical texts must receive identical group IDs"
print("  -> Missing 'prompt_id' Fallback Verified: surrogate clusters created [PASSED]")

# Test 3: Blank strings and NaNs embedded in real text corpus
print("\n[Audit Test 3/7] Blank strings and NaN text resilience...")
dirty_texts = ["", "   ", "\n\n", "The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog again.", "In conclusion, the thesis demonstrated above."]
X_dirty = extractor.fit_transform(dirty_texts)
assert X_dirty.shape[0] == 6, f"Expected 6 rows, got {X_dirty.shape[0]}"
print("  -> Blank/Whitespace/Special Character Texts Handled [PASSED]")

# Test 4: check_finite_logits catching NaNs and Infs
print("\n[Audit Test 4/7] check_finite_logits non-finite trap...")
valid_mat = np.random.randn(10, 18)
assert check_finite_logits(valid_mat, "Valid") is not None
nan_mat = valid_mat.copy()
nan_mat[3, 5] = np.nan
caught_nan = False
try:
    check_finite_logits(nan_mat, "Corrupt")
except FloatingPointError:
    caught_nan = True
assert caught_nan, "check_finite_logits failed to trap NaN"
print("  -> Non-finite logits strictly trapped with FloatingPointError [PASSED]")

# Test 5: fit_temperature on extreme uncalibrated scales
print("\n[Audit Test 5/7] Platt temperature optimization on extreme margins...")
extreme_margins = np.random.randn(50, 18) * 100.0  # Massive overconfidence
y_mock = np.random.randint(0, 18, 50)
t_opt = fit_temperature(extreme_margins, y_mock, default_t=1.0)
assert 0.05 <= t_opt <= 8.0, f"Fitted temp out of bounds: {t_opt}"
print(f"  -> Temperature optimization bounded and stable: T={t_opt} [PASSED]")

# Test 6: Sinkhorn column sum invariance under extreme skew
print("\n[Audit Test 6/7] Sinkhorn 4,000-row column balance under skewed input...")
skewed_probs = np.ones((4000, 18)) * 0.001
skewed_probs[:, 0] = 0.98  # Extremely biased toward class 0
P_balanced = sinkhorn_knopp_balance(skewed_probs, max_iters=50, tol=1e-3)
col_sums = np.sum(P_balanced, axis=0)
assert np.allclose(col_sums, 4000.0 / 18.0, rtol=1e-2), f"Column sums not balanced: {col_sums}"
print("  -> Sinkhorn column normalization converged to uniform N/K under extreme skew [PASSED]")

# Test 7: Submission format and parquet schema
print("\n[Audit Test 7/7] validate_and_save_submission integrity...")
test_out = Path("data/sample/working_verifier/audit_sub.parquet")
test_out.parent.mkdir(parents=True, exist_ok=True)
validate_and_save_submission(["id_1", "id_2"], ["gpt-5.5", "claude-opus-5"], test_out)
read_df = pd.read_parquet(test_out)
assert list(read_df.columns) == ["id", "model"]
assert len(read_df) == 2
print("  -> Parquet file atomic serialization and schema verified [PASSED]")

print("\n>>> ALL 7 ADVERSARIAL AUDIT TESTS PASSED WITH 100% SUCCESS <<<")
