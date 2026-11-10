with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    text = f.read()

target = """def sinkhorn_knopp_balance(
    probs: np.ndarray, max_iters: int = 50, tol: float = 1e-3
) -> np.ndarray:
    \"\"\"
    Sinkhorn-Knopp matrix balancing directly on probability distribution.
    Guarantees uniform column sums (N/K) while maintaining row stochasticity.
    Uses relative tolerance convergence check.
    \"\"\"
    N, K = probs.shape"""

repl = """def sinkhorn_knopp_balance(
    probs: np.ndarray, max_iters: int = 50, tol: float = 1e-3
) -> np.ndarray:
    \"\"\"
    Sinkhorn-Knopp matrix balancing directly on probability distribution.
    Guarantees uniform column sums (N/K) while maintaining row stochasticity.
    Uses relative tolerance convergence check.
    Automatically converts raw margins to calibrated probabilities if negative values are present.
    \"\"\"
    if np.any(probs < 0.0):
        probs = margins_to_probs(probs)
    N, K = probs.shape"""

text_lf = text.replace("\r\n", "\n")
target_lf = target.replace("\r\n", "\n")
repl_lf = repl.replace("\r\n", "\n")

if target_lf in text_lf:
    text_lf = text_lf.replace(target_lf, repl_lf, 1)
    with open("notebooks/kaggle_tri_ensemble.py", "w", encoding="utf-8") as f:
        f.write(text_lf)
    print("Added non-negative probability guard to sinkhorn_knopp_balance in kaggle_tri_ensemble.py")
else:
    print("Target not found in kaggle_tri_ensemble.py!")

# Also update scripts/update_dry_run.py
with open("scripts/update_dry_run.py", "r", encoding="utf-8") as f:
    dry_update = f.read()

dry_target = """    # Verify 4,000-row Sinkhorn matrix balancing
    scores_mock = np.random.randn(4000, 18) * 2.0
    P_mock = sinkhorn_knopp_balance(scores_mock, max_iters=50, tol=1e-3)"""

dry_repl = """    # Verify 4,000-row Sinkhorn matrix balancing
    scores_mock = np.random.randn(4000, 18) * 2.0
    probs_mock = margins_to_probs(scores_mock)
    P_mock = sinkhorn_knopp_balance(probs_mock, max_iters=50, tol=1e-3)"""

dry_update_lf = dry_update.replace("\r\n", "\n")
dry_target_lf = dry_target.replace("\r\n", "\n")
dry_repl_lf = dry_repl.replace("\r\n", "\n")

if dry_target_lf in dry_update_lf:
    dry_update_lf = dry_update_lf.replace(dry_target_lf, dry_repl_lf, 1)
    with open("scripts/update_dry_run.py", "w", encoding="utf-8") as f:
        f.write(dry_update_lf)
    print("Updated scripts/update_dry_run.py with probs_mock!")
else:
    print("dry_target not found in scripts/update_dry_run.py!")

