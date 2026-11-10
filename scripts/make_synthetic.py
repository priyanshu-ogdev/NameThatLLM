"""Generate deterministic integration fixtures. NOT real competition data."""
import argparse
from pathlib import Path

import pandas as pd

from ntl.data import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for prompt in range(15):
        for model_i, model in enumerate(LABELS):
            rows.append({"id": f"synthetic-{prompt}-{model_i}", "prompt_id": str(prompt), "model": model,
                "text": (f"Synthetic topic {prompt}. Author marker{model_i} uses signature{model_i}; "
                         "This fixture checks code and has no scientific value.\n" * (4 + prompt % 4))})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "train.csv", index=False)
    df[["id", "text"]].iloc[:18].to_csv(args.out / "test.csv", index=False)
    (args.out / "SYNTHETIC.txt").write_text("Integration fixture only. Scores do not estimate competition performance.\n")


if __name__ == "__main__":
    main()
