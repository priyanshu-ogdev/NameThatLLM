"""
Comprehensive score projection: what approach combinations can realistically reach 0.70?
All scoring is based on empirical measurements from the 3 gates.

Empirical anchors:
- CPU Dual Linear (C=1.5): ~37.56% on competition rows (measured)
- Oracle Hungarian: +5.60% to +6.15% lift (measured, p<0.0001) ON CLEAN PROMPTS
- Unsupervised Hungarian: -0.04% to -1.53% across all thresholds (measured, Gate 3)
- Confidence-Gated: best was -0.04% at (cos>=0.85, size>=6): almost zero (measured)
- Gate 3 base accuracy (CPU only, 1183 rows): 63.95%
- The test set has ~4000 rows but prompt sizes follow a skewed distribution (most prompts: 1 passage)
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("COMPREHENSIVE AUDIT: PATHS TO 0.70+")
print("=" * 70)

# 1. First, understand the true competition test structure
# The train has mostly 1-passage prompts and a few larger ones
# The test set is likely FULLY STRUCTURED: 4000/18 ≈ 222 prompts × 18 models each
# (or some other structured layout)

# From competition description: test has ~4000 rows across ~222 prompts
# Each prompt was sent to ALL 18 models → each prompt has exactly 18 rows
# That's why we have 18 classes to predict

print("\n=== SECTION 1: TEST SET STRUCTURE REALITY CHECK ===")
print("Train set: 20,583 rows with many singleton prompts (8,484 prompts with only 1 passage)")
print("BUT: Competition test set is likely structured as: N_prompts × 18 models")
print("If test has 4,000 rows: 4000/18 = 222.2 (not integer → incomplete last prompt)")
print("If test has 3,996 rows: 3996/18 = 222 exactly")
print()
print("KEY INSIGHT: The TRAIN data is a mix of:")
print("  - Singletons (one model answered one prompt for annotator variety)")
print("  - Full 18-way groups (the actual competition structure)")
print("  - Partial groups (some models didn't answer certain prompts)")
print()

# 2. Simulate what scores are achievable
print("=== SECTION 2: SCORE PROJECTIONS BY APPROACH ===")
print()

# Empirically measured scores
# CPU alone = 37.56% (production competition set)
# Gate 3 CPU base = 63.95% (but this was on CLEAN prompts with 3+ passages, biased upward)
# True competition CPU estimate is closer to 40-50%
# Gate 1 Oracle = +5.60%/+6.15% (on clean prompts)

approaches = {
    "A: CPU Dual-Linear Only (baseline)": {
        "measured": "37.56% (competition rows), 63.95% (clean multi-model prompts)",
        "test_projection": "~38-45%",
        "confidence": "HIGH - directly measured",
        "risk": "Low",
        "notes": "Baseline. TF-IDF sparse features only."
    },
    "B: + DeBERTa-v3-large fine-tuned": {
        "measured": "Prior audit: 42-48% isolated; ensemble lifts CPU by +3-5%",
        "test_projection": "~44-52% (ensemble)",
        "confidence": "MEDIUM - neural not yet measured on competition test",
        "risk": "Medium (OOM, NaN)",
        "notes": "Transformer adds semantic understanding. LoRA or full fine-tune."
    },
    "C: + BGE-large fine-tuned (sentence embeddings)": {
        "measured": "BGE dense: 70.64% prompt clustering purity (Gate 2), ARI 0.52",
        "test_projection": "+2-4% over DeBERTa blend",
        "confidence": "MEDIUM",
        "risk": "Medium",
        "notes": "Adds representation diversity. Fine-tuned BGE > frozen."
    },
    "D: + Qwen 2.5 7B LoRA": {
        "measured": "Prior: 46-52% isolated; +1-3% blend improvement measured at 40.93%",
        "test_projection": "~50-58% (all 4 streams)",
        "confidence": "MEDIUM - Qwen blend measured at +3.4% over best member",
        "risk": "High (VRAM, FP16 NaN)",
        "notes": "Most powerful per-stream but riskiest."
    },
    "E: Oracle Hungarian (Gate 1) - ONLY POSSIBLE WITH TRUE PROMPT IDs": {
        "measured": "+5.60% to +6.15% on clean prompts (p<0.0001)",
        "test_projection": "+4-6% IF prompt IDs perfectly known",
        "confidence": "HIGH (if prompt IDs known), IMPOSSIBLE (test set - IDs unknown)",
        "risk": "STRUCTURAL: Cannot be deployed without ground truth prompt IDs",
        "notes": "**NOT deployable**. Test set has no prompt IDs."
    },
    "F: Unsupervised Hungarian (Gate 3) - BGE Clustering": {
        "measured": "-0.04% to -1.53% across ALL thresholds (Gate 3 measured)",
        "test_projection": "-0 to -2% (net degradation or zero gain)",
        "confidence": "HIGH (NEGATIVE) - measured, not simulated",
        "risk": "NEGATIVE. Statistically proven to degrade score.",
        "notes": "**REJECT.** Never positive in end-to-end test."
    },
}

for name, info in approaches.items():
    print(f"  [{name}]")
    print(f"    Measured: {info['measured']}")
    print(f"    Projection: {info['test_projection']}")
    print(f"    Confidence: {info['confidence']}")
    print(f"    Risk: {info['risk']}")
    print(f"    Notes: {info['notes']}")
    print()

print("=" * 70)
print("=== SECTION 3: QUANTITATIVE SCORE CEILING ANALYSIS ===")
print()
print("Key question: Is 0.70 reachable with pure independent passage inference?")
print()
print("From the public leaderboard:")
print("  - Top gold medal (Alex Chilton): 0.654")
print("  - Our production target: 4-stream ensemble on dual T4")
print()
print("Gap analysis:")
print("  If best public = 0.654, and we match it: 0.654")
print("  With better transformer fine-tuning: potentially +0.02 to +0.04")
print("  With family-aware auxiliary loss: potentially +0.01 to +0.03")
print("  Combined ceiling (independent inference): ~0.68-0.72")
print()
print("The Oracle Hungarian gate showed +5.6% on the BASE CLASSIFIER.")
print("If the base ensemble reaches 0.64 and prompt IDs were magically known:")
print("  0.64 + 0.056 = 0.696 → approaches 0.70")
print("But without prompt IDs: 0.64 + 0.00 = 0.64")
print()

print("=" * 70)
print("=== SECTION 4: THE ONE REALISTIC PATH TO 0.70 ===")
print()
print("STEP 1: Fix base classifier quality to 0.55+ per-model stream")
print("  - DeBERTa-v3-large: 4 epochs, cosine LR, LLRD, family aux head")
print("  - BGE-large fine-tuned (not frozen): 3 epochs, higher LR for CLS adapter")
print("  - CPU features: preserve C=1.5, add n-gram overlap features for siblings")
print()
print("STEP 2: OOF-fitted blend weights (not 0.85/0.15 hand-set)")
print("  - Fit Simplex blend on OOF via scipy.optimize.minimize")
print("  - This can recover +1-2% over hand-set weights")
print()
print("STEP 3: Family-aware auxiliary loss (multi-task)")
print("  - 6 families: {claude, gemini, gpt, glm, qwen, kimi}")  
print("  - Add auxiliary head: 6-class family loss with weight 0.3")
print("  - Forces representations to first separate families, then siblings")
print()
print("STEP 4: Test-time augmentation (within budget)")
print("  - Predict on passage both forward and in reverse (flip sentences)")
print("  - Average probabilities → removes position-specific stylometric noise")
print()
print("PROJECTED SCORE WITH ALL 4 STEPS:")
print("  Realistic: 0.60 - 0.68 (Central: ~0.64)")
print("  Optimistic: 0.68 - 0.72 (requires all steps to work well)")
print()

print("=" * 70)
print("=== SECTION 5: CAN PROMPT GROUPING HELP AT ALL? ===")
print()
print("The Gate 3 result was net-negative, but was measured on TRAIN data")
print("where prompt groups are mostly singletons (8,484/10,305 prompts = 82% singletons).")
print()
print("On the COMPETITION TEST SET the structure may be different:")
print("  - If test = 222 full 18-model prompts: ALL passages have siblings")
print("  - That's a fundamentally different (better) structure for sibling decoding")
print()
print("However:")
print("  - We still need to RECOVER the prompt groupings without labels")
print("  - Gate 2 showed BGE clustering ARI = 0.52 (far from perfect)")
print("  - Gate 3 on MIXED train data (mostly singletons) degraded")
print("  - ON A FULLY STRUCTURED test (all 18-member prompts), the same BGE")
print("    embedding discrimination (AUC=0.9877) might cluster much better")
print()
print("Recommendation:")
print("  - Measure Gate 2 purity ONLY on prompts with 18 members from train")
print("  - If purity >> 90% in the all-18 subset → sibling decoding becomes viable")
print("  - If still <90% → reject permanently")

