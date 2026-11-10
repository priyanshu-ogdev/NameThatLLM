"""Validate received Kaggle reports and compare paired held-out predictions.

No classifier is trained here. The fixed equal-rank blend is a diagnostic, not a
weight search. Raw IDs, predictions and margins remain in ignored runs/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from ntl.generative import CPU_OOF_SCORE, FOLD_SHA256, TRAIN_SHA256, sha256_file, write_json
from ntl.combined import rank_blend


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--received", type=Path, default=Path("runs/qwen-kaggle-v1-received-20261023"))
    parser.add_argument("--cpu", type=Path, default=Path("runs/linear-midstyle-v1/multi_c150_s020"))
    parser.add_argument("--train", type=Path, default=Path("data/official/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("runs/official-v1/folds.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/qwen-frozen-v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.received / "summary.json").read_text())
    require(summary["training_sha256"] == sha256_file(args.train) == TRAIN_SHA256, "Training hash mismatch")
    require(summary["fold_sha256"] == sha256_file(args.folds) == FOLD_SHA256, "Fold hash mismatch")
    train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
    folds = pd.read_csv(args.folds, dtype={"id": str})
    cpu = pd.read_csv(args.cpu / "oof.csv", dtype={"id": str, "prompt_id": str})
    qwen = pd.read_csv(args.received / "oof.csv", dtype={"id": str})
    for frame in (train, folds, cpu, qwen):
        require(len(frame) == 2699 and frame.id.is_unique, "Invalid row count or duplicate IDs")
        require(frame.id.tolist() == train.id.tolist(), "OOF row order differs from original training data")
    require(qwen.truth.tolist() == cpu.model.tolist() == train.model.tolist(), "OOF truth mismatch")
    require(np.array_equal(qwen.fold, cpu.fold) and np.array_equal(qwen.fold, folds.fold), "OOF fold mismatch")
    require(cpu.prompt_id.tolist() == train.prompt_id.tolist(), "Prompt alignment mismatch")
    require(pd.DataFrame({"prompt": train.prompt_id, "fold": qwen.fold}).groupby("prompt").fold.nunique().max() == 1,
            "Prompt leakage")
    labels = np.array(sorted(train.model.unique()))
    y = train.model.to_numpy()
    cpu_prediction, qwen_prediction = cpu.prediction.to_numpy(), qwen.prediction.to_numpy()
    for prediction in (cpu_prediction, qwen_prediction):
        require(set(prediction) <= set(labels), "Invalid prediction labels")
    cpu_score = float(balanced_accuracy_score(y, cpu_prediction))
    qwen_score = float(balanced_accuracy_score(y, qwen_prediction))
    require(np.isclose(cpu_score, CPU_OOF_SCORE, atol=1e-12, rtol=0), "CPU score mismatch")
    require(np.isclose(qwen_score, summary["winner"]["balanced_accuracy"], atol=1e-12, rtol=0), "Qwen score mismatch")
    per_class = pd.read_csv(args.received / "per_class.csv").set_index("model").loc[labels]
    require(np.allclose(per_class.recall, recall_score(y, qwen_prediction, labels=labels, average=None), atol=1e-12, rtol=0),
            "Reported class recalls mismatch")
    confusion = pd.read_csv(args.received / "confusion.csv", index_col=0).loc[labels, labels].to_numpy()
    require(np.array_equal(confusion, confusion_matrix(y, qwen_prediction, labels=labels)), "Reported confusion mismatch")
    comparison = pd.read_csv(args.received / "comparison.csv")
    require(len(comparison) == 8 and comparison.candidate.is_unique, "Expected eight distinct candidates")
    for candidate in summary["candidates"]:
        row = comparison.set_index("candidate").loc[candidate["candidate"]]
        require(np.isclose(row.balanced_accuracy, candidate["balanced_accuracy"], atol=1e-12, rtol=0), "Candidate summary mismatch")
    require(comparison.loc[comparison.balanced_accuracy.idxmax(), "candidate"] == summary["winner"]["candidate"], "Winner mismatch")
    qwen_margins = np.load(args.received / "decision_scores.npy", allow_pickle=False)
    with np.load(args.cpu / "oof_scores.npz", allow_pickle=False) as archived:
        require(archived["ids"].tolist() == train.id.tolist(), "CPU margin row alignment mismatch")
        order = [archived["labels"].tolist().index(label) for label in labels]
        cpu_margins = archived["scores"][:, order]
    for matrix in (cpu_margins, qwen_margins):
        require(matrix.shape == (2699, 18) and np.isfinite(matrix).all(), "Invalid decision scores")

    # Predefined equal rank averaging avoids mixing incomparable raw score scales.
    # On exact rank-sum ties prefer the CPU decision score; no labels tune this rule.
    blend_prediction = labels[rank_blend(cpu_margins, qwen_margins)]
    blend_score = float(balanced_accuracy_score(y, blend_prediction))
    cpu_correct, qwen_correct = cpu_prediction == y, qwen_prediction == y
    outcome = {"both_correct": int(sum(cpu_correct & qwen_correct)),
               "cpu_only_correct": int(sum(cpu_correct & ~qwen_correct)),
               "qwen_only_correct": int(sum(~cpu_correct & qwen_correct)),
               "neither_correct": int(sum(~cpu_correct & ~qwen_correct))}
    class_table = pd.DataFrame({"model": labels,
                               "cpu_recall": recall_score(y, cpu_prediction, labels=labels, average=None),
                               "qwen_recall": recall_score(y, qwen_prediction, labels=labels, average=None),
                               "fixed_rank_blend_recall": recall_score(y, blend_prediction, labels=labels, average=None)})
    class_table["qwen_minus_cpu"] = class_table.qwen_recall - class_table.cpu_recall
    class_table.to_csv(args.out / "paired_class_recall.csv", index=False)
    fold_rows = []
    predictions = {"cpu": cpu_prediction, "qwen": qwen_prediction, "fixed_rank_blend": blend_prediction}
    for fold in sorted(qwen.fold.unique()):
        mask = qwen.fold.to_numpy() == fold
        fold_rows.append({"fold": int(fold), **{name: float(balanced_accuracy_score(y[mask], pred[mask])) for name, pred in predictions.items()}})
    pd.DataFrame(fold_rows).to_csv(args.out / "paired_fold_scores.csv", index=False)
    words = train.text.str.split().str.len().to_numpy()
    slices = []
    for lower, upper in ((0, 80), (80, 120), (120, 100000)):
        mask = (words >= lower) & (words < upper)
        slices.append({"min_words": lower, "max_words_exclusive": upper, "rows": int(mask.sum()),
                       "represented_classes": len(set(y[mask])),
                       **{name: float(balanced_accuracy_score(y[mask], pred[mask])) for name, pred in predictions.items()}})
    pd.DataFrame(slices).to_csv(args.out / "paired_length_scores.csv", index=False)
    pairs = []
    for true_index, true in enumerate(labels):
        for predicted_index, predicted in enumerate(labels):
            if true != predicted and confusion[true_index, predicted_index]:
                pairs.append({"truth": true, "prediction": predicted, "count": int(confusion[true_index, predicted_index])})
    pd.DataFrame(pairs).sort_values("count", ascending=False).head(20).to_csv(args.out / "top_confused_pairs.csv", index=False)

    # Paired prompt bootstrap: conditional on selected models and fixed OOF fits.
    prompt_codes, unique_prompts = pd.factorize(train.prompt_id, sort=True)
    label_codes = pd.Categorical(y, categories=labels).codes
    total = np.zeros((len(unique_prompts), len(labels)))
    np.add.at(total, (prompt_codes, label_codes), 1)
    weights = np.random.default_rng(2026).multinomial(len(unique_prompts), np.full(len(unique_prompts), 1 / len(unique_prompts)), size=2000)
    denominator = weights @ total
    boot = {}
    for name, pred in predictions.items():
        correct = np.zeros_like(total)
        np.add.at(correct, (prompt_codes, label_codes), (pred == y).astype(int))
        boot[name] = ((weights @ correct) / denominator).mean(axis=1)
    intervals = {name + "_minus_cpu": np.quantile(boot[name] - boot["cpu"], [0.025, 0.975]).tolist()
                 for name in ("qwen", "fixed_rank_blend")}
    report = {"validated": {"rows": len(train), "classes": len(labels), "prompts": len(unique_prompts),
                            "original_data_and_fold_hashes": True, "oof_truth_and_order": True,
                            "confusion_and_class_recall_reconstructed": True, "finite_decision_scores": True,
                            "candidate_summary_consistent": True},
              "balanced_accuracy": {"cpu": cpu_score, "qwen": qwen_score, "fixed_equal_rank_blend_diagnostic": blend_score},
              "paired_correctness_counts": outcome,
              "qwen_correct_on_cpu_errors_fraction": outcome["qwen_only_correct"] / int(sum(~cpu_correct)),
              "prediction_agreement_fraction": float(np.mean(cpu_prediction == qwen_prediction)),
              "qwen_decision_argmax_vs_predict_agreement": float(np.mean(labels[qwen_margins.argmax(axis=1)] == qwen_prediction)),
              "qwen_decision_argmax_balanced_accuracy": float(balanced_accuracy_score(y, labels[qwen_margins.argmax(axis=1)])),
              "conditional_paired_prompt_bootstrap_95pct": intervals, "bootstrap_seed": 2026, "bootstrap_repetitions": 2000,
              "bootstrap_limitations": "Conditional on preselected candidates and existing OOF models; excludes model-selection optimism and refitting uncertainty.",
              "blend_rule": "Equal per-row ranks of 18 decision scores; break rank-sum ties by CPU decision score. No label-fitted weights, thresholds or calibration.",
              "blend_limitations": "Exploratory fixed blend of already selected models on reused folds; not an independent holdout or a leaderboard score. No deployment or full-model blend runtime test yet.",
              "runtime": {"total_seconds": summary["total_seconds"], "train_extraction_seconds": summary["train_extraction"]["seconds"],
                          "classifier_4000_seconds": summary["head_4000_seconds"], "training_truncated_rows": summary["train_extraction"]["truncated_rows"]},
              "provenance": summary["provenance"]}
    write_json(args.out / "paired_analysis.json", report)
    print(json.dumps(report, indent=2))
    print("\nClass comparison:\n", class_table.to_string(index=False))


if __name__ == "__main__":
    main()
