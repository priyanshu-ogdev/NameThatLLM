"""Offline frozen Qwen experiment; torch is optional until extraction is called.

Only pretrained, unchanged representations are shared across folds. Every learned
classifier and scaler is fitted on that fold's training rows alone.
"""
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


MODEL_REPO = "Qwen/Qwen3-1.7B-Base"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
WEIGHT_SHA256 = "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841"
TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLD_SHA256 = "d019f60a2cfaab61dcbb73f64a8c126b0871ff037c23142ef45b81c4cdab6804"
CPU_OOF_SCORE = 0.37560278399204566
# Fold number for each sorted unique prompt_id: no competition IDs or text.
FOLD_PATTERN = [4,4,0,1,4,4,1,3,2,4,0,4,2,3,2,2,4,2,2,0,4,3,1,4,2,3,3,3,2,1,0,1,0,2,1,3,3,2,3,1,3,1,0,4,4,4,4,4,4,2,2,2,0,3,0,4,1,4,2,4,0,1,0,4,2,3,3,1,4,3,0,0,4,1,1,4,0,0,4,3,4,1,0,3,1,2,3,0,0,1,3,0,3,0,0,2,1,0,1,3,3,4,4,2,2,2,1,1,2,0,1,1,2,1,2,3,4,3,3,4,2,2,0,1,2,0,1,2,4,1,1,0,0,2,4,0,0,3,3,0,3,3,1,3,0,1,1,2,2,3]
VIEWS = ("middle_mean", "upper_mean", "final_mean", "final_token")


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def official_folds(train, train_path):
    if sha256_file(train_path) != TRAIN_SHA256:
        raise ValueError("Training data changed; regenerate and review fixed folds before comparing scores.")
    prompts = sorted(train.prompt_id.unique())
    if len(prompts) != len(FOLD_PATTERN):
        raise ValueError("Unexpected prompt count")
    folds = train.prompt_id.map(dict(zip(prompts, FOLD_PATTERN))).to_numpy(dtype=int)
    serialized = pd.DataFrame({"id": train.id, "fold": folds}).to_csv(index=False, lineterminator="\n")
    if hashlib.sha256(serialized.encode()).hexdigest() != FOLD_SHA256:
        raise ValueError("Fold reconstruction differs from the original baseline")
    # Enforce exact duplicate isolation, including duplicates linked across prompts.
    if pd.DataFrame({"text": train.text, "fold": folds}).groupby("text").fold.nunique().max() != 1:
        raise ValueError("Duplicate text crosses validation folds")
    return folds


def check_frame(frame, columns):
    if frame.empty or not set(columns) <= set(frame.columns):
        raise ValueError("Empty input or missing required columns")
    if frame.id.duplicated().any():
        raise ValueError("Duplicate passage IDs")
    for column in columns:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Invalid or blank {column}")


def discover_data(input_root):
    """Prefer official mounts; find an unambiguous train/test pair elsewhere."""
    root = Path(input_root)
    for path in (root / "competitions/name-that-llm", root / "name-that-llm"):
        if (path / "train.csv").is_file() and (path / "test.csv").is_file():
            return path
    matches = sorted({path.parent for path in root.rglob("train.csv")
                      if (path.parent / "test.csv").is_file()})
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError("Multiple train.csv/test.csv pairs found. Remove duplicate data inputs so the intended test file is unambiguous: "
                           + ", ".join(str(path) for path in matches))
    visible = sorted(str(path.relative_to(root)) for path in root.rglob("*.csv"))[:20]
    raise RuntimeError("Attach the Name That LLM competition data using Add Input → Competitions → Name That LLM → Add. "
                       "The notebook needs train.csv and test.csv together. Visible CSV files: "
                       + (", ".join(visible) if visible else "none"))


def discover_model(input_root):
    matches = []
    for config in Path(input_root).rglob("config.json"):
        try:
            data = json.loads(config.read_text())
        except (OSError, ValueError):
            continue
        if (data.get("model_type") == "qwen3" and data.get("hidden_size") == 2048
                and data.get("num_hidden_layers") == 28 and "1.7b-base" in str(config).lower()):
            matches.append(config.parent)
    if len(matches) != 1:
        raise RuntimeError("Attach exactly one official qwen-lm/qwen-3/transformers/1.7b-base/1 model using Add Input → Models.")
    return matches[0]


def load_frozen(model_dir, allow_cpu=False):
    import torch
    from transformers import AutoModel, AutoTokenizer, CONFIG_MAPPING

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("Enable a free GPU in Kaggle Settings → Accelerator, then restart Run All. CPU extraction is disabled to protect your time.")
    if "qwen3" not in CONFIG_MAPPING:
        raise RuntimeError("This runtime needs Transformers >=4.51 with Qwen3 support. Share this error and package version so offline dependencies can be prepared.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device != "cpu" else torch.float32
    if device != "cpu":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}; free {free/2**30:.1f}/{total/2**30:.1f} GiB; dtype {dtype}", flush=True)
        if free < 6 * 2**30:
            raise RuntimeError("At least 6 GiB free GPU memory is required; stop other GPU work and restart the session.")
    model_dir = Path(model_dir)
    if sha256_file(model_dir / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("Model weight checksum differs from the pinned official Qwen3-1.7B-Base version")
    resources = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        resources[name] = sha256_file(model_dir / name)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    version = tuple(int(part) for part in importlib.metadata.version('transformers').split('.')[:2])
    dtype_option = {'dtype': dtype} if version >= (4, 56) else {'torch_dtype': dtype}
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True, attn_implementation="sdpa", **dtype_option)
    model.requires_grad_(False).eval().to(device)
    provenance = {"repo": MODEL_REPO, "upstream_weight_revision": MODEL_REVISION,
                  "source": "qwen-lm/qwen-3/transformers/1.7b-base/1", "license": "Apache-2.0",
                  "weight_sha256": WEIGHT_SHA256, "resource_sha256": resources,
                  "dtype": str(dtype), "device": device, "seed": 42,
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("torch", "transformers", "numpy", "pandas", "scikit-learn", "safetensors", "pyarrow")}}
    return model, tokenizer, provenance


def pool_hidden(hidden_states, attention_mask, layers):
    """Masked means and final actual token, valid for either padding side."""
    import torch

    mask = attention_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("Cannot pool an empty sequence")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask, -1).max(dim=1).values
    rows = torch.arange(len(mask), device=mask.device)
    vectors = []
    for layer in layers:
        state = hidden_states[layer].float()
        local_mask = mask.to(state.device)
        vectors.append(((state * local_mask.unsqueeze(-1)).sum(dim=1) /
                        local_mask.sum(dim=1, keepdim=True)).to(mask.device))
    final = hidden_states[layers[-1]]
    vectors.append(final[rows.to(final.device), last.to(final.device)].float().to(mask.device))
    return torch.stack([torch.nn.functional.normalize(vector, dim=1) for vector in vectors], dim=1)


def extract_frozen(frame, model, tokenizer, provenance, cache_root, max_length=512, batch_size=2, max_seconds=None):
    """Resume flushed batches, keyed by text, IDs, weights, settings and versions."""
    import torch

    if max_length < 2 or batch_size < 1:
        raise ValueError("Invalid token limit or batch size")
    layers = (model.config.num_hidden_layers // 2, model.config.num_hidden_layers * 3 // 4,
              model.config.num_hidden_layers)
    signature = {"schema": 1, "model": provenance, "layers": layers, "max_length": max_length,
                 "batch_size": batch_size, "add_special_tokens": False, "pool": list(VIEWS),
                 "rows_sha256": hashlib.sha256(frame[["id", "text"]].to_json(orient="split", index=False).encode()).hexdigest()}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_root) / key
    cache.mkdir(parents=True, exist_ok=True)
    shape = (len(frame), len(VIEWS), model.config.hidden_size)
    if (cache / "metadata.json").exists():
        if json.loads((cache / "metadata.json").read_text()) != json.loads(json.dumps(signature)):
            raise ValueError("Feature cache metadata differs from the requested extraction")
        vectors = np.lib.format.open_memmap(cache / "vectors.npy", mode="r+")
        done = np.lib.format.open_memmap(cache / "done.npy", mode="r+")
        if vectors.shape != shape or vectors.dtype != np.float32 or done.shape != (len(frame),) or done.dtype != bool:
            raise ValueError("Invalid cache shape/dtype; use a new cache directory")
        if not np.isfinite(vectors[done]).all():
            raise ValueError("Corrupt completed feature cache; use a new cache directory")
        if (cache / "complete.json").exists():
            expected = json.loads((cache / "complete.json").read_text())
            if not done.all() or sha256_file(cache / "vectors.npy") != expected["vectors_sha256"]:
                raise ValueError("Feature cache checksum mismatch; use a new cache directory")
    else:
        vectors = np.lib.format.open_memmap(cache / "vectors.npy", mode="w+", dtype=np.float32, shape=shape)
        done = np.lib.format.open_memmap(cache / "done.npy", mode="w+", dtype=bool, shape=(len(frame),))
        done[:] = False
        vectors.flush(); done.flush()
        write_json(cache / "metadata.json", signature)
    tokens = tokenizer(frame.text.tolist(), add_special_tokens=False, padding=False, truncation=False)
    lengths = np.array([len(ids) for ids in tokens["input_ids"]])
    print(f"Tokens: median {np.median(lengths):.0f}, maximum {lengths.max()}, truncated {sum(lengths > max_length)}/{len(frame)} at {max_length}.", flush=True)
    pending = np.flatnonzero(~done)
    pending = pending[np.argsort(-lengths[pending], kind="stable")]
    started = time.perf_counter()
    for start in range(0, len(pending), batch_size):
        if max_seconds is not None and time.perf_counter() - started >= max_seconds:
            raise TimeoutError('Extraction budget reached; completed batches are saved. Share the pilot/progress report before continuing.')
        indices = pending[start:start + batch_size]
        batch = tokenizer([frame.text.iloc[int(i)] for i in indices], add_special_tokens=False,
                          padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_device = model.get_input_embeddings().weight.device
        batch = {key: value.to(input_device) for key, value in batch.items()}
        with torch.inference_mode():
            result = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True)
            pooled = pool_hidden(result.hidden_states, batch["attention_mask"], layers).cpu().numpy()
        if not np.isfinite(pooled).all():
            raise ValueError("Nonfinite transformer features")
        vectors[indices] = pooled
        vectors.flush()  # Persist vectors before marking their rows complete.
        done[indices] = True
        done.flush()
        del result, batch, pooled
        completed = min(start + batch_size, len(pending))
        if start == 0 or completed % 100 < batch_size or completed == len(pending):
            elapsed = time.perf_counter() - started
            estimate = elapsed / completed * (len(pending) - completed)
            print(f"Features {int(done.sum())}/{len(frame)}; {elapsed/60:.1f} min elapsed; rough remaining {estimate/60:.1f} min", flush=True)
    if not done.all():
        raise RuntimeError("Incomplete feature extraction")
    write_json(cache / "complete.json", {"vectors_sha256": sha256_file(cache / "vectors.npy")})
    return np.asarray(vectors), {"cache_key": key, "rows": len(frame), "seconds": time.perf_counter() - started,
                                 "resumed_rows": len(frame) - len(pending), "tokens_max": int(lengths.max()),
                                 "truncated_rows": int(sum(lengths > max_length)), "max_length": max_length}


def make_head(kernel):
    return make_pipeline(StandardScaler(), Normalizer(),
                         SVC(kernel=kernel, C=1.0 if kernel == "linear" else 10.0,
                             class_weight="balanced", cache_size=512, random_state=42))


def evaluate_heads(features, train, folds, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = np.array(sorted(train.model.unique()))
    y = train.model.to_numpy()
    scores, candidates = [], {}
    for view_index, view in enumerate(VIEWS):
        for kernel in ("linear", "rbf"):
            name = f"{view}_{kernel}"
            started = time.perf_counter()
            predictions = np.empty(len(train), dtype=object)
            decisions = np.zeros((len(train), len(labels)), dtype=np.float32)
            fold_scores = []
            for fold in sorted(set(folds)):
                fitting, held = folds != fold, folds == fold
                if (set(y[fitting]) != set(labels) or set(y[held]) != set(labels)
                        or set(train.prompt_id[fitting]) & set(train.prompt_id[held])):
                    raise ValueError("Invalid grouped folds or missing classes")
                head = make_head(kernel)
                with threadpool_limits(limits=2):
                    head.fit(features[fitting, view_index], y[fitting])
                    predictions[held] = head.predict(features[held, view_index])
                    decisions[held] = head.decision_function(features[held, view_index])
                fold_scores.append(float(balanced_accuracy_score(y[held], predictions[held])))
            score = float(balanced_accuracy_score(y, predictions))
            row = {"candidate": name, "view_index": view_index, "kernel": kernel,
                   "balanced_accuracy": score, "fold_mean": float(np.mean(fold_scores)),
                   "fold_std": float(np.std(fold_scores)), "seconds": time.perf_counter() - started}
            scores.append(row)
            candidates[name] = row
            target = out / name
            target.mkdir(exist_ok=True)
            pd.DataFrame({"id": train.id, "fold": folds, "truth": y, "prediction": predictions}).to_csv(target / "oof.csv", index=False)
            np.save(target / "decision_scores.npy", decisions)
            pd.DataFrame({"model": labels, "recall": recall_score(y, predictions, labels=labels, average=None)}).to_csv(target / "per_class.csv", index=False)
            pd.DataFrame(confusion_matrix(y, predictions, labels=labels), index=labels, columns=labels).to_csv(target / "confusion.csv")
            words = train.text.str.split().str.len().to_numpy()
            slices = []
            for lower, upper in ((0, 80), (80, 120), (120, 100000)):
                mask = (words >= lower) & (words < upper)
                slices.append({"min_words": lower, "max_words_exclusive": upper, "rows": int(mask.sum()),
                               "represented_classes": len(set(y[mask])),
                               "balanced_accuracy": float(balanced_accuracy_score(y[mask], predictions[mask])) if mask.any() else None})
            write_json(target / "metrics.json", {**row, "fold_scores": fold_scores, "labels": labels.tolist(), "word_slices": slices,
                                                 "settings": {"scaler": "StandardScaler", "normalization": "l2", "kernel": kernel,
                                                              "C": 1.0 if kernel == "linear" else 10.0, "gamma": "scale",
                                                              "class_weight": "balanced", "seed": 42}})
            pd.DataFrame(scores).sort_values("balanced_accuracy", ascending=False).to_csv(out / "comparison.csv", index=False)
            print(f"{name}: grouped balanced accuracy {score:.4f}; folds {np.round(fold_scores, 4)}", flush=True)
    winner = max(scores, key=lambda row: row["balanced_accuracy"])
    return winner, scores


def write_submission(test, predictions, labels, work):
    submission = pd.DataFrame({"id": test.id.to_numpy(), "model": predictions})
    if (len(submission) != len(test) or submission.id.duplicated().any()
            or submission.isna().any().any() or not set(submission.model) <= set(labels)):
        raise ValueError("Submission schema, IDs or labels are invalid")
    target = Path(work) / "submission.parquet"
    temporary = target.with_suffix(".parquet.part")
    submission.to_parquet(temporary, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(temporary), submission)
    temporary.replace(target)
    return target


def run_kaggle():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a free GPU using Settings → Accelerator before running this frozen-transformer notebook.")
    started = time.perf_counter()
    work = Path("/kaggle/working")
    out = work / "qwen-frozen-v1"
    out.mkdir(parents=True, exist_ok=True)
    data = discover_data("/kaggle/input")
    print("Competition data found:", data, flush=True)
    train = pd.read_csv(data / "train.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(data / "test.csv", dtype=str, keep_default_na=False)
    check_frame(train, ["id", "prompt_id", "text", "model"])
    check_frame(test, ["id", "text"])
    if train.model.nunique() != 18:
        raise ValueError("Expected 18 model labels")
    folds = official_folds(train, data / "train.csv")
    model_dir = discover_model("/kaggle/input")
    print("Fixed original five prompt-grouped folds verified. Loading", model_dir, flush=True)
    model, tokenizer, provenance = load_frozen(model_dir)
    write_json(out / "provenance.json", provenance)
    features, train_info = extract_frozen(train, model, tokenizer, provenance, out / "cache")
    winner, scores = evaluate_heads(features, train, folds, out)
    head = make_head(winner["kernel"])
    with threadpool_limits(limits=2):
        head.fit(features[:, winner["view_index"]], train.model)
    joblib.dump({"classifier": head, "view_index": winner["view_index"], "provenance": provenance}, out / "classifier.joblib")
    # Head timing at full test scale; feature extraction is measured separately.
    representative = features[np.arange(4000) % len(train), winner["view_index"]]
    benchmark_started = time.perf_counter()
    with threadpool_limits(limits=2):
        head.predict(representative)
    head_4000_seconds = time.perf_counter() - benchmark_started
    print(f"Classifier-only 4000-row timing: {head_4000_seconds:.1f} seconds (not an accuracy test).", flush=True)
    test_features, test_info = extract_frozen(test, model, tokenizer, provenance, out / "cache")
    with threadpool_limits(limits=2):
        predictions = head.predict(test_features[:, winner["view_index"]])
    submission = write_submission(test, predictions, train.model.unique(), work)
    summary = {"winner": winner, "candidates": scores, "cpu_baseline_oof": CPU_OOF_SCORE,
               "cpu_baseline_public_user_reported": 0.372, "training_sha256": TRAIN_SHA256,
               "fold_sha256": FOLD_SHA256, "train_extraction": train_info, "test_extraction": test_info,
               "provenance": provenance, "total_seconds": time.perf_counter() - started,
               "head_4000_seconds": head_4000_seconds,
               "submission_rows": len(test), "test_sha256": sha256_file(data / "test.csv"),
               "selection_note": "Best of eight configurations on existing validation folds; not an unbiased estimate or a leaderboard score."}
    write_json(out / "summary.json", summary)
    print("\nRESULT TO SHARE:", json.dumps({"best_candidate": winner["candidate"],
          "grouped_balanced_accuracy": winner["balanced_accuracy"], "old_cpu_grouped_score": CPU_OOF_SCORE,
          "elapsed_minutes": summary["total_seconds"] / 60, "gpu": torch.cuda.get_device_name(0)}), flush=True)
    print(f"SUCCESS: {submission} with {len(test)} rows. Review comparison.csv before deciding to submit.", flush=True)
