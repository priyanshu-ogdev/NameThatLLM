"""Check train-prompt subspace removal on cached frozen embeddings."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from ntl.cli import fold_input, metadata
from ntl.data import LABELS, fingerprint, read_data, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    df = read_data(args.train)
    folds, _ = fold_input(df, args.folds, args.train)
    source = json.loads((args.embeddings / "manifest.json").read_text())
    if source["data_sha256"] != fingerprint(args.train) or source["fold_sha256"] != fingerprint(args.folds):
        raise ValueError("Embedding data/folds do not match")
    path = args.embeddings / "embeddings.npz"
    if fingerprint(path) != source["embedding_sha256"]:
        raise ValueError("Embedding cache checksum mismatch")
    vectors = np.load(path)
    if not np.array_equal(vectors["ids"], df.id.to_numpy()):
        raise ValueError("Embedding rows do not match training IDs")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "manifest.json", {**metadata(), "data_sha256": fingerprint(args.train),
        "fold_sha256": fingerprint(args.folds), "embedding_sha256": fingerprint(path),
        "note": "Prompt means, nuisance directions and scaling fit only on each outer training partition. Inference projection uses text embeddings only."})
    results = []
    for pooling in ("cls", "mean"):
        matrix = vectors[pooling].astype(np.float64)
        for rank in (16, 48, 96):
            name = f"bge_{pooling}_residual{rank}_rbf10"
            oof = np.empty(len(df), dtype=object)
            fold_scores = []
            start = time.perf_counter()
            for fold in sorted(set(folds)):
                tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                with threadpool_limits(limits=2):
                    mean = matrix[tr].mean(axis=0)
                    group_mean = pd.DataFrame(matrix[tr]).groupby(df.prompt_id.iloc[tr].to_numpy()).mean().to_numpy() - mean
                    _, _, vh = np.linalg.svd(group_mean, full_matrices=False)
                    directions = vh[:rank]
                    train = matrix[tr] - mean
                    valid = matrix[va] - mean
                    train -= (train @ directions.T) @ directions
                    valid -= (valid @ directions.T) @ directions
                    model = make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced"))
                    model.fit(train, df.model.iloc[tr])
                    oof[va] = model.predict(valid)
                fold_scores.append(float(balanced_accuracy_score(df.model.iloc[va], oof[va])))
            out = args.out / name
            out.mkdir()
            df[["id", "prompt_id", "model"]].assign(fold=folds, prediction=oof).to_csv(out / "oof.csv", index=False)
            result = {"candidate": name, "oof_balanced_accuracy": float(balanced_accuracy_score(df.model, oof)),
                      "fold_scores": fold_scores, "seconds": time.perf_counter() - start}
            write_json(out / "metrics.json", result)
            print(json.dumps(result), flush=True)
            results.append(result)
            pd.DataFrame(results).drop(columns="fold_scores").sort_values("oof_balanced_accuracy", ascending=False).to_csv(args.out / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
