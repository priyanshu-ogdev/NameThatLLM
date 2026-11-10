"""Production-grade Kaggle submission validation and Parquet export."""
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from ntl.data import LABELS


def validate_submission_frame(
    submission: pd.DataFrame,
    expected_ids: List[str],
    allowed_labels: List[str] = LABELS
) -> None:
    """
    Strict Kaggle submission schema and integrity verification:
    1. Exactly ['id', 'model'] columns.
    2. Exactly matches expected test IDs in exact order.
    3. Zero nulls, NaNs, or empty strings.
    4. Every prediction is an exact match for one of the 18 official model names.
    """
    if list(submission.columns) != ["id", "model"]:
        raise ValueError(f"Submission columns must be exactly ['id', 'model'], got {list(submission.columns)}")

    if len(submission) != len(expected_ids):
        raise ValueError(f"Submission length mismatch: expected {len(expected_ids)} rows, got {len(submission)}")

    if submission["id"].tolist() != list(expected_ids):
        raise ValueError("Submission IDs do not match expected test IDs in exact order")

    if submission.isna().any().any():
        null_counts = submission.isna().sum().to_dict()
        raise ValueError(f"Submission contains null or NaN values: {null_counts}")

    invalid_predictions = set(submission["model"]) - set(allowed_labels)
    if invalid_predictions:
        raise ValueError(f"Found invalid predicted class labels: {invalid_predictions}")


def export_submission_parquet(
    ids: Union[List[str], np.ndarray, pd.Series],
    predictions: Union[List[str], np.ndarray, pd.Series],
    output_path: Union[str, Path] = "/kaggle/working/submission.parquet"
) -> pd.DataFrame:
    """
    Validates and exports final predictions to Apache Parquet format.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "id": [str(x) for x in ids],
        "model": [str(m) for m in predictions]
    })

    validate_submission_frame(df, expected_ids=[str(x) for x in ids])
    df.to_parquet(output_path, index=False)

    print(f"SUCCESS: Validated submission saved to {output_path.resolve()}")
    print(f"Total rows: {len(df):,}")
    print("Class frequency summary:")
    print(df["model"].value_counts())
    return df

