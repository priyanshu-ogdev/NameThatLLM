"""
Adversarial Failure-Gap & Dry-Run Parity Regression Suite.
Tests production functions directly from notebooks/kaggle_tri_ensemble.py for:
  1. NaN/empty prompt_id DSU clustering safety (no KeyError: nan)
  2. Monotone / <2-class IntraFamilyCliqueSpecialist safety
  3. quad_rank_blend & optimize_blend_weights with svm=None / partial OOF masks
  4. Real tiny HuggingFace encoder & Qwen+PEFT precompute_token_caches + cached_enc_* split slice + Fold 1->2 retain_backbone & exception cleanup
  5. run_quad_ensemble Step 3/4 simulation with empty test_*_folds, missing SVM, and uncertain test_probs[0] preservation
  6. dry_run_verifier_template AST check ensuring smoke_caches uses split slices (tr_slice/va_slice) matching texts lengths
"""
import gc
import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "notebooks" / "kaggle_tri_ensemble.py"
DRY_RUN_PATH = ROOT / "notebooks" / "kaggle_dry_run.py"

spec = importlib.util.spec_from_file_location("kte", str(SCRIPT_PATH))
kte = importlib.util.module_from_spec(spec)
sys.modules["kte"] = kte
spec.loader.exec_module(kte)

LABELS = kte.LABELS
NUM_CLASSES = kte.NUM_CLASSES


def test_1_prompt_groups_nan_and_empty_ids():
    df = pd.DataFrame({
        "prompt_id": [np.nan, np.nan, "", "", "p1", None],
        "text": ["same body", "unique body 1", "unique body 2", "unique body 3", "unique body 4", "same body"],
    })
    groups = kte.make_prompt_groups(df)
    assert len(groups) == 6
    # Identical texts must merge via DSU; distinct NaN/empty rows must stay separate
    assert groups[0] == groups[5], "Duplicate text with NaN/None prompt_ids must union"
    assert len({groups[1], groups[2], groups[3], groups[4]}) == 4, "Distinct NaN/empty prompt_ids must not collide"
    print("[PASS] Test 1: make_prompt_groups NaN/None/empty prompt_id safety")


def test_2_single_class_clique_and_tiny_subspace():
    X = sp.csr_matrix(np.random.RandomState(0).randn(30, 16).astype(np.float32))
    y_mono = np.array([LABELS[0]] * 30)
    spec_clf = kte.IntraFamilyCliqueSpecialist().fit(X, y_mono)
    assert len(spec_clf.specialists_) == 0, "Single-class clique must be skipped safely"

    V_single = np.random.randn(1, 20).astype(np.float32)
    proj = kte.StyleInvariantSubspaceProjector().fit(V_single, np.array(["p0"]), y_labels=np.array([LABELS[0]]))
    out = proj.transform(V_single)
    assert out.shape == (1, 20) and np.all(np.isfinite(out))

    V_nan = np.array([[np.nan] * 5, [1.0] * 5], dtype=np.float32)
    proj2 = kte.StyleInvariantSubspaceProjector().fit(V_nan, np.array(["p0", "p0"]), y_labels=np.array([LABELS[0], LABELS[1]]))
    assert np.all(np.isfinite(proj2.transform(V_nan)))
    print("[PASS] Test 2: Single-class clique & tiny/NaN StyleInvariantSubspaceProjector")


def test_3_blend_and_quad_rank_with_none_svm():
    rng = np.random.RandomState(42)
    n_oof, n_te = 60, 20
    y_int = np.arange(n_oof) % NUM_CLASSES
    y_str = np.array([LABELS[i] for i in y_int])
    oof_deb = rng.randn(n_oof, NUM_CLASSES) * 0.5
    for i in range(n_oof):
        oof_deb[i, y_int[i]] += 1.5
    oof_deb[:10, :] = np.nan  # Simulate partial missing Fold 1

    best_w, best_alpha, score = kte.optimize_blend_weights(None, oof_deb, None, None, y_str)
    assert best_w[0] == 0.0 and best_score_valid(score)
    preds = kte.quad_rank_blend(None, rng.randn(n_te, NUM_CLASSES), weights=best_w, balance_test=False)
    assert len(preds) == n_te and all(0 <= p < NUM_CLASSES for p in preds)
    print(f"[PASS] Test 3: optimize_blend_weights & quad_rank_blend with svm=None (score={score:.4f})")


def best_score_valid(s):
    return np.isfinite(s) and 0.0 <= s <= 1.0


def test_4_dry_run_cache_slicing_and_tiny_hf_fold_lifecycle():
    # Inspect dry_run source to guarantee smoke_caches uses tr_slice / va_slice (no len=4 vs len=40 mismatch)
    dry_text = DRY_RUN_PATH.read_text(encoding="utf-8")
    assert 'm_cache["tr_slice"]' in dry_text and 'm_cache["va_slice"]' in dry_text
    assert 'precompute_token_caches(texts_tr[:4], texts_te[:4]' not in dry_text

    # Build real tiny local BERT & Qwen2 models on disk and test precompute_token_caches -> tr_slice -> va_slice -> train_*_fold
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
        Qwen2Config,
        Qwen2Model,
        Qwen2TokenizerFast,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        deb_dir = tmp_path / "tiny_bert"
        deb_dir.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "test", "text", "model"]
        (deb_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
        tok_b = BertTokenizerFast(vocab_file=str(deb_dir / "vocab.txt"))
        tok_b.save_pretrained(deb_dir)
        cfg_b = BertConfig(
            vocab_size=len(vocab), hidden_size=32, num_hidden_layers=4,
            num_attention_heads=2, intermediate_size=64, num_labels=NUM_CLASSES,
        )
        BertForSequenceClassification(cfg_b).save_pretrained(deb_dir)

        texts_tr = ["hello world test text model"] * 8
        texts_va = ["test text hello model"] * 4
        texts_te = ["world hello text"] * 3
        y_tr = [LABELS[i % 4] for i in range(8)]
        y_va = [LABELS[i % 4] for i in range(4)]

        caches = kte.precompute_token_caches(texts_tr + texts_va, texts_te, deberta_dir=str(deb_dir), max_length=32)
        deb_c = caches["deberta"]
        tr_slice = {k: deb_c["tr"][k][:8] for k in deb_c["tr"]}
        va_slice = {k: deb_c["tr"][k][8:12] for k in deb_c["tr"]}

        val_l, te_l = kte.train_bge_fold(
            texts_tr, y_tr, texts_va, y_va, texts_te,
            str(deb_dir), LABELS,
            device="cpu", batch_size=4, grad_accum=1, max_length=32, epochs=2,
            enable_grad_checkpointing=False, n_frozen_bottom_layers=1,
            cached_enc_tr=tr_slice, cached_enc_va=va_slice, cached_enc_te=deb_c["te"],
            cached_enc_te_tta=deb_c["te_tta"], cached_tokenizer=deb_c["tokenizer"],
            group_ids_tr=np.array([f"p{i%3}" for i in range(8)]),
        )
        assert val_l.shape == (4, NUM_CLASSES) and np.all(np.isfinite(val_l))
        assert te_l.shape == (3, NUM_CLASSES) and np.all(np.isfinite(te_l))

        # Test tiny Qwen2 Fold 1 (retain_backbone=True) -> Fold 2 cache reuse & exception-safe cache offload
        q_dir = tmp_path / "tiny_qwen"
        q_dir.mkdir()
        q_cfg = Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        )
        Qwen2Model(q_cfg).save_pretrained(q_dir)
        from tokenizers import Tokenizer, models, pre_tokenizers
        tk_json = Tokenizer(models.WordLevel(vocab={"<endoftext>": 0, "<unk>": 1, "hello": 2, "world": 3, "test": 4}, unk_token="<unk>"))
        tk_json.pre_tokenizer = pre_tokenizers.Whitespace()
        tok_q = Qwen2TokenizerFast(tokenizer_object=tk_json, eos_token="<endoftext>", pad_token="<endoftext>")
        tok_q.save_pretrained(q_dir)

        qc = kte.precompute_token_caches(texts_tr + texts_va, texts_te, qwen_dir=str(q_dir), max_length=32)["qwen"]
        q_tr_slice = {k: qc["tr"][k][:8] for k in qc["tr"]}
        q_va_slice = {k: qc["tr"][k][8:12] for k in qc["tr"]}

        val_q1, te_q1 = kte.train_qwen_lora_fold(
            texts_tr, y_tr, texts_va, y_va, texts_te,
            str(q_dir), LABELS,
            batch_size=4, grad_accum=1, max_length=32, epochs=2, lora_r=4, lora_alpha=8,
            retain_backbone=True,
            cached_enc_tr=q_tr_slice, cached_enc_va=q_va_slice, cached_enc_te=qc["te"],
            cached_enc_te_tta=qc["te_tta"], cached_tokenizer=qc["tokenizer"],
        )
        assert val_q1.shape == (4, NUM_CLASSES) and te_q1.shape == (3, NUM_CLASSES)
        assert "qwen_backbone" in kte.GLOBAL_RAM_MODEL_CACHE

        # Fold 2 reuses RAM cache
        val_q2, te_q2 = kte.train_qwen_lora_fold(
            texts_tr, y_tr, texts_va, y_va, texts_te,
            str(q_dir), LABELS,
            batch_size=4, grad_accum=1, max_length=32, epochs=1, lora_r=4, lora_alpha=8,
            retain_backbone=False,
            cached_enc_tr=q_tr_slice, cached_enc_va=q_va_slice, cached_enc_te=qc["te"],
            cached_enc_te_tta=qc["te_tta"], cached_tokenizer=qc["tokenizer"],
        )
        assert val_q2.shape == (4, NUM_CLASSES) and te_q2.shape == (3, NUM_CLASSES)
        assert "qwen_backbone" not in kte.GLOBAL_RAM_MODEL_CACHE

        # Clean up GLOBAL_RAM_MODEL_CACHE from test
        kte.GLOBAL_RAM_MODEL_CACHE.clear()
        gc.collect()
    print("[PASS] Test 4: Tiny HF Encoder + Qwen2 Fold 1->2 retain_backbone & TTA cached_enc_* lifecycle")


def test_5_run_quad_ensemble_partial_and_na_ids():
    # End-to-end run_quad_ensemble on tiny synthetic CSV including string id="NA", NaN prompt_id, and partial neural failures
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        work_dir = tmp_path / "work"
        data_dir.mkdir()
        work_dir.mkdir()

        rng = np.random.RandomState(123)
        n_tr = 90
        tr_rows = []
        for i in range(n_tr):
            cls = LABELS[i % NUM_CLASSES]
            p_idx = i // NUM_CLASSES
            pid = np.nan if i == 0 else f"prompt_{p_idx}"
            tr_rows.append({"id": f"tr_{i}", "prompt_id": pid, "text": f"style token {cls} variant {i} body text here.", "model": cls})
        pd.DataFrame(tr_rows).to_csv(data_dir / "train.csv", index=False)

        te_rows = [{"id": "NA" if i == 0 else f"te_{i}", "text": f"style token {LABELS[i % NUM_CLASSES]} body."} for i in range(12)]
        pd.DataFrame(te_rows).to_csv(data_dir / "test.csv", index=False)

        # 5a. Simulate DeBERTa producing valid val_deb on folds while test_deb=None (empty test_deb_folds)
        def _deb_val_only(texts_tr, y_tr_str, texts_va, y_va_str, *args, **kwargs):
            v = rng.randn(len(y_va_str), NUM_CLASSES) * 0.2
            for row_i, lbl in enumerate(y_va_str):
                v[row_i, kte.LABEL_TO_IDX[lbl]] += 2.0
            return v, None

        with mock.patch.object(kte, "discover_paths", return_value=(data_dir, None, tmp_path / "fake_deb", None, None, work_dir)):
            with mock.patch.object(kte, "precompute_token_caches", return_value={}):
                with mock.patch.object(kte, "train_deberta_fold", side_effect=_deb_val_only):
                    kte.run_quad_ensemble()

        sub_a = pd.read_parquet(work_dir / "submission.parquet")
        assert len(sub_a) == 12 and sub_a["id"].iloc[0] == "NA"

        # 5b. Real CPU-only run_quad_ensemble with string id="NA" & NaN prompt_ids
        with mock.patch.object(kte, "discover_paths", return_value=(data_dir, None, None, None, None, work_dir)):
            kte.run_quad_ensemble()

        sub = pd.read_parquet(work_dir / "submission.parquet")
        assert len(sub) == 12
        assert sub["id"].iloc[0] == "NA", f"String 'NA' test id was mangled to {sub['id'].iloc[0]!r}"
        assert sub["model"].isin(LABELS).all()
    print("[PASS] Test 5: run_quad_ensemble end-to-end with string id='NA' & NaN prompt_id")


def test_6_external_discovery_and_schema_resilience():
    import os
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        comp_dir = tmp_path / "competitions" / "name-that-llm"
        ext_dir = tmp_path / "datasets" / "owner" / "ext-pack"
        work_dir = tmp_path / "work"
        comp_dir.mkdir(parents=True)
        ext_dir.mkdir(parents=True)
        work_dir.mkdir(parents=True)

        # Competition CSVs + distracting incompatible parquets
        comp_df = pd.DataFrame([
            {"id": f"c_{i}", "prompt_id": f"cp_{i // NUM_CLASSES}", "text": f"comp {LABELS[i % NUM_CLASSES]} {i}", "model": LABELS[i % NUM_CLASSES]}
            for i in range(36)
        ])
        comp_df.to_csv(comp_dir / "train.csv", index=False)
        pd.DataFrame([{"id": f"t_{i}", "text": "test row"} for i in range(5)]).to_csv(comp_dir / "test.csv", index=False)
        pd.DataFrame([{"id": "x", "wrong_col": 1}]).to_parquet(tmp_path / "external_junk.parquet", index=False)

        # Wide external WITHOUT 'id' column (uses auto-id)
        wide_path = ext_dir / "custom_multi_llm.parquet"
        wide_row = {lbl: f"wide text {lbl}" for lbl in LABELS}
        pd.DataFrame([wide_row, wide_row]).to_parquet(wide_path, index=False)

        # Wide external CSV fallback check
        wide_csv_path = ext_dir / "external_wide_fallback.csv"
        pd.DataFrame([wide_row, wide_row]).to_csv(wide_csv_path, index=False)

        # Simulate huggingfacedebertav3variants with both deberta-v3-large and deberta-v3-large-offensive
        deb_off = tmp_path / "datasets" / "huggingfacedebertav3variants" / "deberta-v3-large-offensive"
        deb_clean = tmp_path / "datasets" / "huggingfacedebertav3variants" / "deberta-v3-large"
        deb_off.mkdir(parents=True)
        deb_clean.mkdir(parents=True)
        (deb_off / "config.json").write_text('{"model_type": "deberta-v2"}', encoding="utf-8")
        (deb_clean / "config.json").write_text('{"model_type": "deberta-v2"}', encoding="utf-8")

        old_env = {k: os.environ.get(k) for k in ("NTL_DATA_DIR", "NTL_WORKING", "NTL_EXT_PATH")}
        try:
            os.environ["NTL_DATA_DIR"] = str(tmp_path)
            os.environ["NTL_WORKING"] = str(work_dir)
            os.environ.pop("NTL_EXT_PATH", None)
            _, _, found_deb, _, found_ext, _ = kte.discover_paths()
            assert found_deb is not None and Path(found_deb) == deb_clean, f"Expected clean {deb_clean}, got {found_deb}"
            assert found_ext is not None and Path(found_ext) == wide_path, f"Expected {wide_path}, got {found_ext}"

            comb_csv, w_csv = kte.build_mass_balanced_external_corpus(comp_df, wide_csv_path, LABELS)
            assert len(comb_csv) == len(comp_df) + 2 * NUM_CLASSES
            assert np.isclose(w_csv.sum(), float(len(comp_df)), atol=1e-4)

            comb_w, w_w = kte.build_mass_balanced_external_corpus(comp_df, found_ext, LABELS)
            assert len(comb_w) == len(comp_df) + 2 * NUM_CLASSES
            assert np.isclose(w_w.sum(), float(len(comp_df)), atol=1e-4)

            # Long external schema override via NTL_EXT_PATH
            long_path = ext_dir / "ext_long_pack.parquet"
            long_df = pd.DataFrame([
                {"prompt_id": f"lp_{i}", "model": LABELS[i % NUM_CLASSES], "text": f"long sample {i}"}
                for i in range(NUM_CLASSES * 2)
            ])
            long_df.to_parquet(long_path, index=False)
            os.environ["NTL_EXT_PATH"] = str(long_path)
            _, _, _, _, found_long, _ = kte.discover_paths()
            assert found_long is not None and Path(found_long) == long_path
            comb_l, w_l = kte.build_mass_balanced_external_corpus(comp_df, found_long, LABELS)
            assert len(comb_l) == len(comp_df) + len(long_df)
            assert np.isclose(w_l.sum(), float(len(comp_df)), atol=1e-4)
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    print("[PASS] Test 6: Nested external discovery, junk rejection & wide/long schema resilience")


if __name__ == "__main__":
    test_1_prompt_groups_nan_and_empty_ids()
    test_2_single_class_clique_and_tiny_subspace()
    test_3_blend_and_quad_rank_with_none_svm()
    test_4_dry_run_cache_slicing_and_tiny_hf_fold_lifecycle()
    test_5_run_quad_ensemble_partial_and_na_ids()
    test_6_external_discovery_and_schema_resilience()
    print("\nALL 6 ADVERSARIAL FAILURE-GAP & PARITY TESTS PASSED!")
