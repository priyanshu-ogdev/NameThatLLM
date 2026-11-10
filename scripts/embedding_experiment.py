"""Frozen, local-only BERT encoder experiment using the project's fixed folds.

Run in the separate embeddings environment. This never updates transformer
parameters, calls a remote API, or uses validation labels to fit a representation.
"""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits

from ntl.cli import fold_input, metadata
from ntl.data import LABELS, fingerprint, read_data, write_json


def extract(texts, model_dir, batch_size=16, max_length=512, threads=2):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                       use_safetensors=True).eval()
    result = {"cls": [], "mean": []}
    lengths = tokenizer(list(texts), truncation=False, add_special_tokens=True)["input_ids"]
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            encoded = tokenizer(list(texts[i:i+batch_size]), return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length)
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            result["cls"].append(torch.nn.functional.normalize(hidden[:, 0], dim=1).numpy())
            result["mean"].append(torch.nn.functional.normalize(pooled, dim=1).numpy())
            if i % (batch_size * 10) == 0:
                print(json.dumps({"encoded": min(i+batch_size, len(texts)), "total": len(texts),
                                  "seconds": time.perf_counter() - start}), flush=True)
    return {name: np.vstack(values) for name, values in result.items()}, {
        "seconds": time.perf_counter() - start, "max_length": max_length,
        "truncated_rows": sum(len(v) > max_length for v in lengths), "batch_size": batch_size,
        "max_observed_tokens": max(map(len, lengths)), "threads": threads,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    df = read_data(args.train)
    folds, _ = fold_input(df, args.folds, args.train)
    args.out.mkdir(parents=True, exist_ok=False)
    vectors, extraction = extract(df.text.to_numpy(), args.encoder)
    np.savez_compressed(args.out / "embeddings.npz", **vectors, ids=df.id.to_numpy(dtype=str))
    write_json(args.out / "manifest.json", {**metadata(), "data_sha256": fingerprint(args.train),
        "fold_sha256": fingerprint(args.folds), "encoder_provenance": json.loads((args.encoder / "provenance.json").read_text()),
        "extraction": extraction, "embedding_sha256": fingerprint(args.out / "embeddings.npz"),
        "extra_versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "safetensors"]}})
    results = []
    for pooling, matrix in vectors.items():
        for method, classifier in [
            ("lr1", LogisticRegression(C=1, class_weight="balanced", max_iter=2500)),
            ("lr10", LogisticRegression(C=10, class_weight="balanced", max_iter=2500)),
            ("svm1", LinearSVC(C=1, class_weight="balanced", max_iter=10000)),
            ("rbf10", SVC(C=10, gamma="scale", class_weight="balanced")),
        ]:
            name = f"bge_{pooling}_{method}"
            oof = np.empty(len(df), dtype=object)
            scores = np.empty((len(df), len(LABELS)))
            per_fold = []
            started = time.perf_counter()
            for f in sorted(set(folds)):
                tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
                # Normalization and classifier fit stay inside the training fold.
                model = make_pipeline(StandardScaler(), classifier)
                with threadpool_limits(limits=2):
                    model.fit(matrix[tr], df.model.iloc[tr])
                    oof[va] = model.predict(matrix[va])
                    raw = model.decision_function(matrix[va])
                scores[va] = raw[:, [list(model.classes_).index(label) for label in LABELS]]
                per_fold.append(float(balanced_accuracy_score(df.model.iloc[va], oof[va])))
            out = args.out / name
            out.mkdir()
            frame = df[["id", "prompt_id", "model"]].assign(fold=folds, prediction=oof)
            frame.to_csv(out / "oof.csv", index=False)
            np.savez_compressed(out / "oof_scores.npz", scores=scores, labels=np.array(LABELS), ids=df.id.to_numpy(dtype=str))
            pd.DataFrame({"model": LABELS, "recall": recall_score(df.model, oof, labels=LABELS, average=None)}).to_csv(out / "per_class_recall.csv", index=False)
            pd.DataFrame(confusion_matrix(df.model, oof, labels=LABELS), index=LABELS, columns=LABELS).to_csv(out / "confusion.csv")
            result = {"candidate": name, "oof_balanced_accuracy": float(balanced_accuracy_score(df.model, oof)),
                      "fold_scores": per_fold, "seconds": time.perf_counter() - started,
                      "note": "Frozen encoder, with classifier/scaler fit only inside each training fold; configuration comparison is model selection."}
            write_json(out / "metrics.json", result)
            print(json.dumps(result), flush=True)
            results.append(result)
            pd.DataFrame(results).drop(columns=["fold_scores", "note"]).sort_values("oof_balanced_accuracy", ascending=False).to_csv(args.out / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
