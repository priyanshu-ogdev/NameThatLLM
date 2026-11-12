"""Commands for auditing, evaluating, training, and offline prediction."""
import argparse
import importlib.metadata
import json
import platform
try:
    import resource
except ImportError:
    resource = None
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits

from ntl.data import LABELS, audit, fingerprint, make_folds, read_data, validate_folds, write_json
from ntl.models import BASELINES, CANDIDATES, build_model
from ntl.ensemble import fit_grouped
from ntl.data import groups


def get_peak_rss_mib():
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / (1024 * 1024)
    except Exception:
        return None


def metadata():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "git_dirty": dirty, "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["name-that-llm", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow"]},
    }


def fold_input(df, path, train_path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["data_sha256"] != fingerprint(train_path):
        raise ValueError("Data hash changed; create a new version of the fold file")
    folds = validate_folds(df, pd.read_csv(path, dtype={"id": str}, keep_default_na=False))
    return folds, meta


def prepare(args):
    df = read_data(args.train)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "audit.json", audit(df))
    folds = make_folds(df, args.folds, args.seed)
    validate_folds(df, folds)
    folds.to_csv(out / "folds.csv", index=False)
    write_json(out / "folds.json", {**metadata(), "data_sha256": fingerprint(args.train),
                                    "n_splits": args.folds, "seed": args.seed,
                                    "method": "StratifiedGroupKFold with duplicate-connected prompt groups"})
    print((out / "audit.json").read_text())


def evaluate(args):
    df = read_data(args.train)
    folds, fold_meta = fold_input(df, args.folds, args.train)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "manifest.json", {**metadata(), "data_sha256": fingerprint(args.train),
        "fold_sha256": fingerprint(args.folds), "fold_metadata": fold_meta, "seed": args.seed,
        "candidates": args.candidates, "threads": args.threads,
        "warning": "Selecting the best candidate on these folds is model selection, not an unbiased final performance estimate."})
    results = []
    for name in args.candidates:
        out = root / name
        out.mkdir()
        oof = np.empty(len(df), dtype=object)
        scores = np.zeros((len(df), len(LABELS)))
        fold_results = []
        started = time.perf_counter()
        model = build_model(name, args.seed)
        write_json(out / "settings.json", {k: repr(v) for k, v in model.get_params(deep=True).items()})
        for f in sorted(set(folds)):
            tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            model = build_model(name, args.seed)
            if args.cache and hasattr(model, "memory"):
                model.set_params(memory=joblib.Memory(args.cache, verbose=0))
            t = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                with threadpool_limits(limits=args.threads):
                    fit_grouped(model, df.text.iloc[tr], df.model.iloc[tr], groups(df)[tr])
                    oof[va] = model.predict(df.text.iloc[va])
                    raw = (model.decision_function(df.text.iloc[va]) if hasattr(model, "decision_function")
                           else model.predict_proba(df.text.iloc[va]))
            order = [list(model.classes_).index(label) for label in LABELS]
            scores[va] = raw[:, order]
            result = {"fold": int(f), "train_rows": len(tr), "valid_rows": len(va),
                      "balanced_accuracy": float(balanced_accuracy_score(df.model.iloc[va], oof[va])),
                      "seconds": time.perf_counter() - t,
                      "warnings": [str(w.message) for w in caught]}
            fold_results.append(result)
            print(json.dumps({"candidate": name, **result}), flush=True)
        prediction = df[["id", "prompt_id", "model"]].copy()
        prediction["fold"] = folds
        prediction["prediction"] = oof
        prediction["word_count"] = df.text.str.split().str.len()
        prediction.to_csv(out / "oof.csv", index=False)
        # These are margins/logits or native probabilities, depending on estimator.
        np.savez_compressed(out / "oof_scores.npz", scores=scores, labels=np.array(LABELS), ids=df.id.to_numpy(dtype=str))
        recall = recall_score(df.model, oof, labels=LABELS, average=None, zero_division=0)
        pd.DataFrame({"model": LABELS, "recall": recall}).to_csv(out / "per_class_recall.csv", index=False)
        pd.DataFrame(confusion_matrix(df.model, oof, labels=LABELS), index=LABELS, columns=LABELS).to_csv(out / "confusion.csv")
        matrix = confusion_matrix(df.model, oof, labels=LABELS)
        pairs = [(LABELS[i], LABELS[j], int(matrix[i,j])) for i in range(18) for j in range(18) if i != j and matrix[i,j] > 0]
        pd.DataFrame(sorted(pairs, key=lambda row: row[2], reverse=True), columns=["true", "predicted", "count"]).to_csv(out / "confused_pairs.csv", index=False)
        length_results = []
        for lo, hi in [(0, 80), (80, 150), (150, 1000000)]:
            ix = (prediction.word_count >= lo) & (prediction.word_count < hi)
            if ix.any():
                recalls = recall_score(df.model[ix], oof[ix], labels=LABELS, average=None, zero_division=0)
                present = np.array([(df.model[ix] == lab).any() for lab in LABELS])
                length_results.append({"min_words": lo, "max_words_exclusive": hi, "rows": int(ix.sum()),
                    "classes_present": int(present.sum()), "balanced_accuracy_present_classes": float(recalls[present].mean())})
        write_json(out / "length_analysis.json", length_results)
        result = {"candidate": name, "oof_balanced_accuracy": float(balanced_accuracy_score(df.model, oof)),
                  "mean_fold_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in fold_results])),
                  "std_fold_balanced_accuracy": float(np.std([r["balanced_accuracy"] for r in fold_results])),
                  "seconds": time.perf_counter() - started,
                  "process_peak_rss_mib": get_peak_rss_mib(),
                  "folds": fold_results}
        write_json(out / "metrics.json", result)
        results.append({k: v for k, v in result.items() if k != "folds"})
        pd.DataFrame(results).sort_values("oof_balanced_accuracy", ascending=False).to_csv(root / "comparison.csv", index=False)
    print(pd.read_csv(root / "comparison.csv").to_string(index=False))


def train(args):
    df = read_data(args.train)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    model = build_model(args.candidate, args.seed)
    t = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        fit_grouped(model, df.text, df.model, groups(df))
    joblib.dump({"pipeline": model, "labels": LABELS}, out / "model.joblib", compress=3)
    write_json(out / "manifest.json", {**metadata(), "candidate": args.candidate, "seed": args.seed,
        "data_sha256": fingerprint(args.train), "rows": len(df), "seconds": time.perf_counter() - t,
        "model_sha256": fingerprint(out / "model.joblib"), "labels": LABELS})
    print(f"Saved {out / 'model.joblib'}")


def predict_frame(bundle, test, batch_size=256):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    predictions = []
    for start in range(0, len(test), batch_size):
        predictions.extend(bundle["pipeline"].predict(test.text.iloc[start:start + batch_size]))
    out = pd.DataFrame({"id": test.id.to_numpy(), "model": predictions})
    validate_submission(out, test, bundle["labels"])
    return out


def validate_submission(out, test, labels=LABELS):
    if list(out.columns) != ["id", "model"] or out.isna().any().any():
        raise ValueError("Submission needs exactly id/model columns without nulls")
    if len(out) != len(test) or out.id.duplicated().any() or set(out.id) != set(test.id):
        raise ValueError("Every test ID must appear exactly once")
    if not set(out.model) <= set(labels):
        raise ValueError("Invalid predicted label")


def load_bundle(path):
    # joblib can execute code: load only artifacts you trained or explicitly trust.
    path = Path(path)
    manifest_paths = [p for p in (path.parent / "manifest.json", path.parent / "training_manifest.json") if p.exists()]
    if len(manifest_paths) != 1:
        raise ValueError("Expected exactly one training manifest alongside the model")
    manifest = json.loads(manifest_paths[0].read_text())
    if fingerprint(path) != manifest["model_sha256"]:
        raise ValueError("Model checksum does not match training manifest")
    for name, expected in manifest["versions"].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Package mismatch for {name}: trained with {expected}, installed {actual}")
    bundle = joblib.load(path)
    if set(bundle["labels"]) != set(LABELS):
        raise ValueError("Artifact labels do not match configured competition labels")
    return bundle


def predict(args):
    test = read_data(args.test, train=False)
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Refusing to overwrite {out}")
    t = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        submission = predict_frame(load_bundle(args.model), test, args.batch_size)
    if args.id_dtype == "int64":
        # Require lossless conversion; leading-zero string IDs must stay strings.
        converted = submission.id.astype("int64")
        if converted.astype(str).tolist() != submission.id.tolist():
            raise ValueError("ID conversion is not lossless; choose string IDs")
        submission["id"] = converted
        test = test.copy()
        test["id"] = converted.to_numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(out, index=False)
    validate_submission(pd.read_parquet(out), test)
    print(json.dumps({"rows": len(test), "seconds": time.perf_counter() - t,
                      "output": str(out), "sha256": fingerprint(out)}))


def benchmark(args):
    test = read_data(args.test, train=False)
    # Repetition tests runtime and memory, not accuracy on the hidden distribution.
    scaled = test.iloc[np.arange(args.rows) % len(test)].copy().reset_index(drop=True)
    scaled["id"] = [f"benchmark-{i}" for i in range(len(scaled))]
    t = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        predict_frame(load_bundle(args.model), scaled, args.batch_size)
    report = {**metadata(), "rows": len(scaled), "seconds": time.perf_counter() - t,
              "process_peak_rss_mib": get_peak_rss_mib(),
              "model_sha256": fingerprint(args.model), "source_sha256": fingerprint(args.test),
              "note": "Repeated available passages; runtime proxy only. Recheck using representative real passage lengths and final artifact."}
    path = Path(args.out)
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    write_json(path, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=prepare)
    p = sub.add_parser("evaluate")
    p.add_argument("--train", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(BASELINES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cache", help="Optional local feature cache; fit cache keys include fold-local text and labels")
    p.set_defaults(func=evaluate)
    p = sub.add_parser("train")
    p.add_argument("--train", required=True)
    p.add_argument("--candidate", choices=CANDIDATES, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(func=train)
    for command, func in [("predict", predict), ("benchmark", benchmark)]:
        p = sub.add_parser(command)
        p.add_argument("--model", required=True)
        p.add_argument("--test", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--threads", type=int, default=2)
        if command == "predict":
            p.add_argument("--id-dtype", choices=("string", "int64"), default="string")
        if command == "benchmark":
            p.add_argument("--rows", type=int, default=4000)
        p.set_defaults(func=func)
    args = parser.parse_args()
    if hasattr(args, "threads") and args.threads < 1:
        parser.error("--threads must be positive")
    if hasattr(args, "rows") and args.rows < 1:
        parser.error("--rows must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
