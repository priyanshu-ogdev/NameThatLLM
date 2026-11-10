"""Real CPU functional check for frozen Qwen and resumable feature extraction.

Uses optional embeddings environment and the verified official weights. This is
an integration check, never an authorship accuracy experiment.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ntl.generative import extract_frozen, load_frozen, pool_hidden, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    hidden = (torch.tensor([[[1., 0.], [0., 2.], [99., 99.]],
                            [[99., 99.], [3., 0.], [0., 4.]]]),)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    result = pool_hidden(hidden, mask, (0, 0, 0))
    np.testing.assert_allclose(result[:, 0], np.array([[1, 2], [3, 4]]) / np.linalg.norm([[1, 2], [3, 4]], axis=1, keepdims=True), atol=1e-6)
    np.testing.assert_allclose(result[:, -1], [[0, 1], [0, 1]])
    model, tokenizer, provenance = load_frozen(args.model, allow_cpu=True)
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    frame = pd.DataFrame({"id": ["001", "002", "003"], "text": [
        "Small foxes run quickly.", "A second paragraph has rather different punctuation: why?",
        "An intentionally longer sentence. " * 30]})
    with tempfile.TemporaryDirectory(prefix="ntl-qwen-check-") as directory:
        vectors, info = extract_frozen(frame, model, tokenizer, provenance, directory, max_length=64, batch_size=2)
        assert vectors.shape == (3, 4, 2048) and np.isfinite(vectors).all()
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=2), 1, atol=1e-5)
        assert info["truncated_rows"] == 1
        original = vectors.copy()
        again, repeated = extract_frozen(frame, model, tokenizer, provenance, directory, max_length=64, batch_size=2)
        assert repeated["resumed_rows"] == 3
        np.testing.assert_array_equal(original, again)
        # Simulate interruption before the final row's completion flag was saved.
        cache = Path(directory) / info["cache_key"]
        (cache / "complete.json").unlink()
        done = np.lib.format.open_memmap(cache / "done.npy", mode="r+")
        done[0] = False
        done.flush()
        resumed, partial = extract_frozen(frame, model, tokenizer, provenance, directory, max_length=64, batch_size=2)
        assert partial["resumed_rows"] == 2
        np.testing.assert_allclose(original, resumed, atol=1e-5)
        # A changed passage must never reuse the old passage's cached representation.
        changed = frame.copy()
        changed.loc[0, "text"] = "A replacement sentence with different vocabulary."
        altered, altered_info = extract_frozen(changed, model, tokenizer, provenance, directory, max_length=64, batch_size=2)
        assert altered_info["cache_key"] != info["cache_key"]
        assert not np.allclose(original[0], altered[0])
        corrupt = np.lib.format.open_memmap(cache / "vectors.npy", mode="r+")
        corrupt[0, 0, 0] += 0.1
        corrupt.flush()
        try:
            extract_frozen(frame, model, tokenizer, provenance, directory, max_length=64, batch_size=2)
        except ValueError as error:
            assert "checksum" in str(error)
        else:
            raise AssertionError("Corrupt cache was accepted")
    report = {"purpose": "CPU integration check, not competition evaluation", "rows": len(frame),
              "shape": list(original.shape), "frozen_parameters": True, "finite_normalized_vectors": True,
              "padding_masks": "left and right verified", "complete_cache_reuse": True,
              "partial_resume": True, "changed_text_cache_isolation": True, "corruption_rejected": True,
              "seconds": time.perf_counter() - started, "provenance": provenance,
              "gpu_execution": "unrun; no GPU in this cloud environment"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
