import ast

with open("notebooks/kaggle_dry_run.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replacement 1: Check 7
old_check7 = """    # Full Competition Score Projection based on 18-class Macro Balanced Accuracy
    base_cpu_score = 0.3556
    lift_tri_style = 0.0280     # Positional Tri-Partition Stylometrics (Head 25%, Body 50%, Tail 25%)
    lift_deb = 0.1450           # DeBERTa-v3-large contextual attention + SWA Checkpoint Averaging
    lift_bge = 0.0720           # BGE-large dense semantic geometry + SWA Checkpoint Averaging
    lift_qwen = 0.0950          # Qwen 2.5 7B LoRA generative likelihood + SWA Averaging
    lift_hierarchical = 0.0350  # Two-Stage Hierarchical Family Posterior Gating with Cardinality Normalization
    lift_sinkhorn = 0.0240      # Platt Temperature Calibration + Sinkhorn-Knopp Prior Balancing
    proj_ensemble_score = base_cpu_score + lift_tri_style + lift_deb + lift_bge + lift_qwen + lift_hierarchical + lift_sinkhorn
    score_low = proj_ensemble_score - 0.020
    score_high = proj_ensemble_score + 0.020

    print("\\n" + "=" * 70)
    print(">>> PROJECTED COMPETITION SCORE MATRIX (18-CLASS BALANCED ACCURACY) <<<")
    print("=" * 70)
    print("  Metric: Macro Balanced Accuracy (Random Baseline = 1/18 = 5.55%)")
    print("-" * 70)
    print(f"  1. CPU Multiview Dual-Linear (Empirical CV):         {base_cpu_score*100:5.2f}%  (6.4x above random chance)")
    print(f"  2. + Positional Tri-Partition Stylometrics:          +{lift_tri_style*100:4.1f}%")
    print(f"  3. + Neural Contextual Attention (DeBERTa FT + SWA): +{lift_deb*100:4.1f}%")
    print(f"  4. + Dense Semantic Geometry (BGE-large FT + SWA):   +{lift_bge*100:4.1f}%")
    print(f"  5. + 7B Generative Likelihood & Flow (Qwen LoRA+SWA):+{lift_qwen*100:4.1f}%")
    print(f"  6. + Two-Stage Hierarchical Family Posterior Gating: +{lift_hierarchical*100:4.1f}%")
    print(f"  7. + Calibration & Sinkhorn-Knopp Prior Balancing:   +{lift_sinkhorn*100:4.1f}%")
    print("-" * 70)
    print(f"  PROJECTED FINAL ENSEMBLE SCORE:  {proj_ensemble_score*100:.2f}%  [{score_low*100:.1f}% - {score_high*100:.1f}%]")
    print("  COMPETITION BENCHMARK CONTEXT:")
    print("    - Random Chance Baseline:        5.55%  (1/18)")
    print("    - Public Char TF-IDF Baseline:  31.40%")
    print("    - Public CPU Multiview SVM:     37.20%")
    print("    - Top Public Solution (Chilton): 65.40%  (Gold Medal Tier)")
    print("    - THIS QUAD-MODEL ENSEMBLE:     75.46%  (Top 1% Frontier Target)")
    print("=" * 70 + "\\n", flush=True)"""

new_check7 = """    # Empirical Benchmark & Real Calibration Summary
    real_cpu_measured = benchmarks["cpu_linear"]["empirical_cv_score"]
    print("\\n" + "=" * 70)
    print(">>> EMPIRICAL BENCHMARK & REAL VALIDATION SUMMARY <<<")
    print("=" * 70)
    print("  Metric: Macro Balanced Accuracy across 18 classes (Random Baseline = 1/18 = 5.55%)")
    print("-" * 70)
    print(f"  1. CPU Multiview Dual-Linear (Empirical CV):         {real_cpu_measured*100:5.2f}%  ({real_cpu_measured/0.05555:.1f}x above random chance)")
    print("  2. Historical 5-Fold CPU Baseline (Full N=2,699):    35.56%  (6.4x above random chance)")
    print(f"  3. Temperature Calibration (Platt NLL Fit):          Verified [OK] (Optimal T={t_opt:.2f})")
    print(f"  4. Universal Quality Gate (<1.5/18 chance filter):   Verified [OK] (chance accuracy {chance_acc*100:.1f}% dropped)")
    print(f"  5. Sinkhorn-Knopp Matrix Balancing (Target 222/cls): Verified [OK] (Min={counts.min()}, Max={counts.max()})")
    print("-" * 70)
    print("  COMPETITION BENCHMARK CONTEXT & REALISTIC TARGET RANGES:")
    print("    - Random Chance Baseline:        5.55%  (1/18)")
    print("    - Single CPU Linear (Measured): 35.56%  (Prompt-grouped 5-fold CV on N=2,699)")
    print("    - Top Public Baseline (Chilton): 65.40%  (Alex Chilton 0.654 milestone)")
    print("    - Full 4-Stream Ensemble Scope:  Mid-50s to Mid-60s (Empirical OOF computed in tri_ensemble)")
    print("=" * 70 + "\\n", flush=True)"""

# Replacement 2: Check 8 footer
old_check8_footer = """    print(f"  PROJECTED COMPETITION SCORE:   {proj_ensemble_score*100:.2f}% (Macro Balanced Accuracy - Top 1% Frontier Target)")"""
new_check8_footer = """    print(f"  EMPIRICAL CPU CV BENCHMARK:     {real_cpu_measured*100:.2f}% (Full ensemble score evaluated via OOF in tri_ensemble)")"""

# Normalize newlines for robust matching
text_crlf = text.replace("\r\n", "\n")
old_check7_norm = old_check7.replace("\r\n", "\n")
new_check7_norm = new_check7.replace("\r\n", "\n")

assert old_check7_norm in text_crlf, "old_check7 not found in dry_run"
assert old_check8_footer in text_crlf, "old_check8_footer not found in dry_run"

text_crlf = text_crlf.replace(old_check7_norm, new_check7_norm, 1)
text_crlf = text_crlf.replace(old_check8_footer, new_check8_footer, 1)

# Verify AST
ast.parse(text_crlf)

with open("notebooks/kaggle_dry_run.py", "w", encoding="utf-8") as f:
    f.write(text_crlf)

print("kaggle_dry_run.py updated with empirical, uninflated validation reporting!")

