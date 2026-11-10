# Measured experiment reports

- `cpu-v1/`: 19 completed configurations, five fixed prompt-grouped folds, seed
  42. `comparison.csv` has scores, timing, memory and conditional prompt-bootstrap
  intervals. Other files describe the selected model, errors and offline checks.
- `frozen-embeddings-v1/`: 14 completed frozen BGE configurations on the same data
  and folds. Includes source/license/checksum and extraction provenance.

These are local validation results, not leaderboard results. Small differences
among the best configurations are within sampling uncertainty. Selection on the
same folds causes optimism; bootstrap intervals do not correct it. Aborted and
unrun configurations have no claimed complete score. Only aggregates and hashes
are published; text, row IDs, weights and OOF arrays remain in ignored local paths.
