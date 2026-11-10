"""Export aggregate results without publishing competition text, IDs, or artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ntl.data import LABELS, fingerprint, write_json


def group_bootstrap(oof, samples=2000, seed=2026):
    frame = oof.assign(correct=(oof.model == oof.prediction).astype(int))
    correct = frame.pivot_table(index="prompt_id", columns="model", values="correct", aggfunc="sum", fill_value=0).reindex(columns=LABELS).to_numpy()
    total = frame.pivot_table(index="prompt_id", columns="model", values="correct", aggfunc="count", fill_value=0).reindex(columns=LABELS).to_numpy()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(correct), size=(samples, len(correct)))
    return (correct[ix].sum(axis=1) / total[ix].sum(axis=1)).mean(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, distributions, sources = [], {}, []
    expected = None
    for root in args.runs:
        manifest = json.loads((root / "manifest.json").read_text())
        signature = (manifest["data_sha256"], manifest["fold_sha256"])
        if expected is not None and expected != signature:
            raise ValueError("Cannot compare different data or folds")
        expected = signature
        sources.append({"run": str(root), "manifest_sha256": fingerprint(root / "manifest.json")})
        for path in sorted(root.glob("*/metrics.json")):
            metrics = json.loads(path.read_text())
            name = metrics["candidate"]
            if name in distributions:
                raise ValueError(f"Repeated candidate {name}; select one explicit version")
            oof = pd.read_csv(path.parent / "oof.csv", dtype={"id": str, "prompt_id": str})
            if oof.prediction.isna().any() or oof.id.duplicated().any():
                raise ValueError("Incomplete or duplicate OOF rows")
            distribution = group_bootstrap(oof)
            distributions[name] = distribution
            rows.append({k: metrics[k] for k in ["candidate", "oof_balanced_accuracy", "mean_fold_balanced_accuracy", "std_fold_balanced_accuracy", "seconds", "process_peak_rss_mib"]} | {
                "prompt_bootstrap_ci95_low": float(np.quantile(distribution, .025)),
                "prompt_bootstrap_ci95_high": float(np.quantile(distribution, .975)),
                "fold_count": len(metrics["folds"]), "run_path": str(path.parent),
                "warnings": sum(len(fold["warnings"]) for fold in metrics["folds"]),
            })
    comparison = pd.DataFrame(rows).sort_values("oof_balanced_accuracy", ascending=False)
    comparison.to_csv(args.out / "comparison.csv", index=False)
    best = comparison.iloc[0]
    best_path = Path(best.run_path)
    # These reports contain class-level aggregates only.
    for name in ["per_class_recall.csv", "confusion.csv", "confused_pairs.csv", "length_analysis.json"]:
        (args.out / name).write_bytes((best_path / name).read_bytes())
    delta = distributions[best.candidate] - distributions["word_lr"]
    write_json(args.out / "summary.json", {"data_sha256": expected[0], "fold_sha256": expected[1],
        "selected_candidate": best.candidate, "oof_balanced_accuracy": float(best.oof_balanced_accuracy),
        "paired_prompt_bootstrap_delta_vs_word_lr_ci95": list(np.quantile(delta, [.025,.975])),
        "bootstrap_seed": 2026, "bootstrap_samples": 2000, "source_manifests": sources,
        "caveat": "Models were compared on these folds: selection optimism remains. Bootstrap intervals resample prompt groups and do not account for model selection, training randomness, or test distribution shift. This is not a leaderboard score. Stacking fits its combiner only on inner grouped OOF predictions."})
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
