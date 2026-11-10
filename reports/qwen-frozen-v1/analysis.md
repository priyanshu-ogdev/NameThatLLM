# Frozen Qwen: received reports and paired analysis

All six user-supplied reports were downloaded and verified on October 23, 2026.
These are real prompt-grouped validation results. No new GPU job was needed.

| Approach | Balanced accuracy on existing held-out predictions |
| --- | ---: |
| Selected CPU TF-IDF/style SVM | 37.56% |
| Frozen Qwen middle-layer mean + linear SVM | 35.12% |
| Fixed equal-rank combination | **40.93%** |

The original training CSV and fold CSV hashes match, and all 2,699 IDs, truths
and fold assignments align with the retained CPU predictions. Reconstructing
Qwen's confusion matrix and class recalls reproduced the supplied reports.
All margins are finite with 18 columns. All eight candidate summaries agree.
No training passages were truncated: maximum 381 tokens with a limit of 512.
Extraction took 577.03 seconds and the whole script 991.49 seconds (16.52 min).
The 4,000-row classifier-only benchmark took 10.26 seconds. This excludes
feature extraction, GPU session startup/idle time and hidden-test evaluation.

The models make different errors:

- Both correct: **531** passages.
- CPU alone correct: **483** passages.
- Qwen alone correct: **417** passages.
- Neither correct: **1,268** passages.

Qwen correctly classifies 24.75% of the CPU model's errors. Relative to CPU,
its class recall is higher for kimi-k3 (+11.33 points), gpt-5.6-luna (+10.00),
claude-opus-4-8 (+8.67), claude-fable-5 (+8.05), glm-5.3-flash (+6.67) and
gpt-5.6-terra (+5.33). It is substantially worse for gemini-3.1-pro-preview
(-28 points), qwen3.8-max (-22), and gpt-6-astra (-16). No single model wins
uniformly. Short passages remain harder for both.

The fixed blend ranks each model's 18 decision scores within a passage, sums
the two ranks with equal weights, and uses CPU decision scores to break exact
ties. This avoids mixing incompatible raw score scales; it does not treat SVM
scores as probabilities. No validation-label fitting, grid search, calibration
or threshold tuning was used for this rule. It improved on CPU in all five
folds and all three length slices. The gain is 3.37 percentage points overall.
The conditional 95% paired-prompt bootstrap interval is +1.71 to +5.07 points
(2,000 resamples, seed 2026).

This is an exploratory diagnostic on reused folds with already-selected base
configurations. The interval excludes model-selection optimism, training-set
resampling/refitting uncertainty and hidden-test distribution shift. It is not
an independent holdout estimate or a Kaggle score. A complete refit/inference
blend must be implemented and runtime-tested before submission. The CPU model
remains the best actual submitted model, with public score 0.372.

Qwen SVC's default multiclass prediction and decision-score argmax differ on
about 3.0% of rows because ties are resolved differently. Score-argmax alone
measures 35.56%; it is not the original 35.12% prediction rule. The blend uses
the stored decision scores with its own explicit tie-breaking rule.

## Retention and reproduction

The six originals are in ignored `runs/qwen-kaggle-v1-received-20261023/`.
A verified separate backup ZIP is in ignored
`artifacts/qwen-kaggle-v1-reports.zip` (203,532 bytes including its receipt
manifest). The manifest retains uploaded file IDs, original names, sizes and
SHA-256 values. These private files depend on workspace/snapshot retention;
Git does not restore them. No permanent attachment-service retention is assumed.

Only safe aggregate reports, package/data hashes and the reproducible analysis
script are committed. `received_files.json` records the public checksums;
`kaggle_summary.json` preserves settings and provenance. Row-level predictions
and margins remain private. The archive was read back and every original's hash
was rechecked before this analysis.

From the project root with retained private inputs:

```bash
/workspace/venvs/name-that-llm/bin/python scripts/analyze_qwen_reports.py
```

See `paired_analysis.json`, `paired_class_recall.csv`, `paired_fold_scores.csv`,
`paired_length_scores.csv`, `top_confused_pairs.csv` and `comparison.csv` here.
No extra spending or paid API calls occurred.
