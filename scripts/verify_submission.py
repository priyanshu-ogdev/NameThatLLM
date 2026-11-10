"""
Automated Kaggle submission linter, schema validator, and offline scorer for Name That LLM.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ntl.data import LABELS


def verify_submission(parquet_path: Path, test_csv_path: Path, solution_csv_path: Path = None):
    print(f"=== AUDITING KAGGLE SUBMISSION: {parquet_path.name} ===")

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Submission file does not exist: {parquet_path}")

    # Check file size
    size_bytes = parquet_path.stat().st_size
    print(f"File Size: {size_bytes:,} bytes")
    if size_bytes == 0:
        raise ValueError("Submission file is empty (0 bytes)")

    sub = pd.read_parquet(parquet_path)
    test = pd.read_csv(test_csv_path, dtype={"id": str})

    # 1. Column verification
    cols = list(sub.columns)
    if cols != ["id", "model"]:
        raise ValueError(f"Invalid columns: expected ['id', 'model'], found {cols}")
    print("Column Names: ['id', 'model'] (Verified)")

    # 2. Row count verification
    if len(sub) != len(test):
        raise ValueError(f"Row count mismatch: test set has {len(test):,} rows, but submission has {len(sub):,} rows")
    print(f"Row Count: {len(sub):,} rows (Matches test.csv exactly)")

    # 3. ID order and uniqueness
    sub_ids = sub["id"].astype(str).tolist()
    test_ids = test["id"].astype(str).tolist()
    if sub_ids != test_ids:
        raise ValueError("Submission IDs do not match test IDs in exact order!")
    if len(set(sub_ids)) != len(sub_ids):
        raise ValueError("Duplicate passage IDs detected in submission!")
    print("ID Integrity: 100% matched in exact order without duplicates")

    # 4. Null & NaN check
    if sub.isna().any().any():
        nulls = sub.isna().sum().to_dict()
        raise ValueError(f"Submission contains null or NaN values: {nulls}")
    print("Null / NaN Check: 0 nulls, 0 empty strings (Passed)")

    # 5. Class label validity
    unique_preds = set(sub["model"].unique())
    invalid = unique_preds - set(LABELS)
    if invalid:
        raise ValueError(f"Illegal class labels detected in predictions: {invalid}")
    print(f"Class Label Validity: All {len(unique_preds)} predicted classes are valid official LABELS")

    # 6. Distribution summary
    counts = sub["model"].value_counts().to_dict()
    print("\nPredicted Class Frequency Summary:")
    for m in LABELS:
        c = counts.get(m, 0)
        pct = (c / len(sub)) * 100
        print(f"  {m:25s}: {c:5d} ({pct:5.1f}%)")

    # 7. Ground truth evaluation (if solution provided)
    score_report = {}
    if solution_csv_path and solution_csv_path.is_file():
        sol = pd.read_csv(solution_csv_path, dtype={"id": str})
        y_true = sol.set_index("id").loc[sub["id"]]["model"].to_numpy()
        y_pred = sub["model"].to_numpy()
        b_acc = balanced_accuracy_score(y_true, y_pred)
        print(f"\n=======================================================")
        print(f"EVALUATION ON GROUND TRUTH: Balanced Accuracy = {b_acc * 100:.2f}%")
        print(f"=======================================================\n")
        score_report["balanced_accuracy"] = float(b_acc)

    report = {
        "submission_path": str(parquet_path),
        "size_bytes": size_bytes,
        "row_count": len(sub),
        "passed_all_checks": True,
        "class_counts": counts,
        **score_report
    }

    report_path = parquet_path.parent / f"{parquet_path.stem}_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Audit report saved to: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit and verify Kaggle submission files.")
    parser.add_argument("--submission", type=Path, default=Path("data/sample/working/submission.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/sample/test.csv"))
    parser.add_argument("--solution", type=Path, default=None)
    args = parser.parse_args()

    verify_submission(args.submission, args.test, args.solution)


if __name__ == "__main__":
    main()

