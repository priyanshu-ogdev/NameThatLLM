"""
Comprehensive dataset auditing, cleaning, prompt-grouped Stratified K-Fold splitting,
and finalized preprocessed asset generation for Name That LLM.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ntl.data import LABELS
from ntl.preprocessing import clean_text, RichStylometricExtractor


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(s: str) -> str:
    return hashlib.sha256(" ".join(s.split()).encode("utf-8")).hexdigest()


class DisjointSetUnion:
    def __init__(self, elements):
        self.parent = {x: x for x in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def audit_corpus(df: pd.DataFrame, stage_name: str) -> dict:
    words = df["text"].apply(lambda t: len(t.split()))
    chars = df["text"].apply(len)
    sentences = df["text"].apply(lambda t: len([s for s in re.split(r"[.!?]+\s*", t) if s.strip()]))
    hashes = df["text"].apply(text_hash)

    # Stylometric markers
    headers = df["text"].str.contains(r"^\s*#{1,6}\s", regex=True)
    code_blocks = df["text"].str.contains("```")
    bullets = df["text"].str.contains(r"^\s*[-*•]\s|^\s*\d+\.\s", regex=True)
    refusals = df["text"].str.contains(r"as an ai language model|under anthropic's usage policy|i cannot fulfill this request|i am unable to provide", regex=True, flags=re.I)

    grouped_hashes = df.assign(thash=hashes).groupby("thash")

    return {
        "stage": stage_name,
        "total_rows": len(df),
        "unique_prompts": int(df["prompt"].nunique()),
        "class_counts": {k: int(v) for k, v in df["model"].value_counts().items()},
        "word_count_quantiles": {str(k): float(v) for k, v in words.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).items()},
        "char_count_quantiles": {str(k): float(v) for k, v in chars.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).items()},
        "sentence_count_quantiles": {str(k): float(v) for k, v in sentences.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).items()},
        "duplicate_text_passages": int(hashes.duplicated(keep=False).sum()),
        "cross_label_duplicates": int((grouped_hashes["model"].nunique() > 1).sum()),
        "passages_with_markdown_headers": int(headers.sum()),
        "passages_with_code_blocks": int(code_blocks.sum()),
        "passages_with_bullet_lists": int(bullets.sum()),
        "passages_matching_refusal_patterns": int(refusals.sum())
    }


def main():
    workspace = Path(__file__).resolve().parent.parent
    raw_path = workspace / "datasets" / "external_short_wide.parquet"
    out_dir = workspace / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw external wide corpus: {raw_path}")
    wide_df = pd.read_parquet(raw_path)
    model_cols = [c for c in wide_df.columns if c not in ["id", "prompt"]]
    assert set(model_cols) == set(LABELS), f"Columns mismatch official LABELS: {set(model_cols) ^ set(LABELS)}"

    # 1. Melt to long format
    print("Melting wide format to long format...")
    raw_records = []
    for _, row in wide_df.iterrows():
        ext_id = row["id"]
        prompt_txt = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
        for m in model_cols:
            ans = row[m]
            if pd.notna(ans) and isinstance(ans, str) and ans.strip():
                raw_records.append({
                    "raw_id": ext_id,
                    "prompt": prompt_txt.strip(),
                    "model": m,
                    "text": ans
                })

    df_raw = pd.DataFrame(raw_records)
    print(f"Extracted {len(df_raw)} raw responses across 18 models.")
    initial_audit = audit_corpus(df_raw, "initial_raw_extraction")

    # 2. Clean and Normalize Text
    print("Applying text cleaning and normalization...")
    df_raw["text"] = df_raw["text"].apply(clean_text)
    df_raw["prompt"] = df_raw["prompt"].apply(clean_text)

    # 3. Handle boilerplate canned refusals and duplicates
    print("Auditing and handling duplicates...")
    # Compute normalized text hashes
    df_raw["text_hash"] = df_raw["text"].apply(text_hash)

    # Detect exact duplicate passages
    # For canned policy refusals that repeat identically across many prompts, keep only 1 canonical instance
    dup_counts = df_raw["text_hash"].value_counts()
    frequent_dups = dup_counts[dup_counts > 1].index.tolist()

    keep_mask = np.ones(len(df_raw), dtype=bool)
    dropped_refusals = 0
    seen_frequent_hashes = set()

    for idx, row in df_raw.iterrows():
        th = row["text_hash"]
        if th in frequent_dups:
            # Check if this is a canned safety refusal
            if "blocked under Anthropic's Usage Policy" in row["text"] or "restrictions on violative cyber content" in row["text"]:
                if th in seen_frequent_hashes:
                    keep_mask[idx] = False
                    dropped_refusals += 1
                else:
                    seen_frequent_hashes.add(th)

    df_clean = df_raw[keep_mask].copy().reset_index(drop=True)
    print(f"Deduplicated {dropped_refusals} redundant canned refusal copies.")

    # 4. Length Filtering
    df_clean["word_count"] = df_clean["text"].apply(lambda t: len(t.split()))
    df_clean["char_count"] = df_clean["text"].apply(len)

    # Verify all passages are within valid competition length bounds [40, 250]
    valid_length = (df_clean["word_count"] >= 40) & (df_clean["word_count"] <= 250)
    df_clean = df_clean[valid_length].copy().reset_index(drop=True)
    print(f"Passages after strict length verification [40, 250]: {len(df_clean)}")

    # 5. Deterministic Prompt Grouping & Canonical IDs
    print("Assigning deterministic prompt IDs and canonical passage IDs...")
    # Group identical prompt texts
    unique_prompts = sorted(df_clean["prompt"].unique())
    prompt_to_id = {p: f"p{i:05d}" for i, p in enumerate(unique_prompts)}
    df_clean["prompt_id"] = df_clean["prompt"].map(prompt_to_id)

    # Assign deterministic canonical IDs
    df_clean["id"] = [f"ntl_{row['prompt_id']}_{row['model']}_{i:03d}" for i, row in df_clean.iterrows()]

    clean_audit = audit_corpus(df_clean, "clean_deduplicated_corpus")

    # 6. Build DSU Duplicate Connected Components
    print("Building Disjoint Set Union (DSU) graph for prompt duplicate isolation...")
    dsu = DisjointSetUnion(df_clean["prompt_id"].unique())
    seen_hashes = {}
    for pid, th in zip(df_clean["prompt_id"], df_clean["text_hash"]):
        if th in seen_hashes:
            dsu.union(pid, seen_hashes[th])
        else:
            seen_hashes[th] = pid

    df_clean["dsu_group"] = df_clean["prompt_id"].apply(dsu.find)
    print(f"Unique prompts: {df_clean['prompt_id'].nunique()} -> Connected DSU groups: {df_clean['dsu_group'].nunique()}")

    # 7. Stratified Partitioning: Holdout Test Set vs Training Set
    # We use a 7-fold StratifiedGroupKFold on DSU groups:
    # 1 fold (~14.3%) becomes the Holdout Test Set.
    # The other 6 folds (~85.7%) form the Training Set.
    print("Partitioning Holdout Test Set (14.3%) and Training Set (85.7%) via StratifiedGroupKFold...")
    splitter_outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    outer_splits = list(splitter_outer.split(df_clean["text"], df_clean["model"], df_clean["dsu_group"]))

    train_idx, test_idx = outer_splits[0]
    df_train = df_clean.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df_clean.iloc[test_idx].copy().reset_index(drop=True)

    # Verify all 18 labels are present in both train and test
    assert set(df_train["model"]) == set(LABELS), f"Train set missing models: {set(LABELS) - set(df_train['model'])}"
    assert set(df_test["model"]) == set(LABELS), f"Test set missing models: {set(LABELS) - set(df_test['model'])}"

    # Verify zero prompt or duplicate leakage between train and test
    train_groups = set(df_train["dsu_group"])
    test_groups = set(df_test["dsu_group"])
    assert len(train_groups & test_groups) == 0, "Catastrophic Leakage: prompt groups overlap between train and test!"

    train_prompts = set(df_train["prompt"])
    test_prompts = set(df_test["prompt"])
    assert len(train_prompts & test_prompts) == 0, "Catastrophic Leakage: prompt text overlaps between train and test!"

    # 8. Create 5-Fold StratifiedGroupKFold on Training Set
    print("Partitioning Training Set into 5 StratifiedGroupKFold cross-validation folds...")
    splitter_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = np.full(len(df_train), -1, dtype=int)

    for fold_num, (tr_i, val_i) in enumerate(splitter_cv.split(df_train["text"], df_train["model"], df_train["dsu_group"])):
        # Verify all 18 labels in each partition
        assert set(df_train["model"].iloc[tr_i]) == set(LABELS), f"CV Fold {fold_num} train missing models"
        assert set(df_train["model"].iloc[val_i]) == set(LABELS), f"CV Fold {fold_num} valid missing models"
        cv_folds[val_i] = fold_num

    df_train["fold"] = cv_folds

    # Verify zero leakage across CV folds
    for f in range(5):
        tr_g = set(df_train[df_train["fold"] != f]["dsu_group"])
        va_g = set(df_train[df_train["fold"] == f]["dsu_group"])
        assert len(tr_g & va_g) == 0, f"Leakage detected in CV fold {f}"

    # Set split label in clean corpus
    df_clean["split"] = "train"
    df_clean.loc[test_idx, "split"] = "test"

    # 9. Format Final Tables
    core_cols = ["id", "prompt_id", "prompt", "model", "text", "word_count", "char_count"]
    df_clean_final = df_clean[core_cols + ["split"]].copy()
    df_train_final = df_train[core_cols + ["fold"]].copy()
    df_test_final = df_test[core_cols].copy()

    # Create unlabelled test format matching Kaggle test.csv
    df_test_unlabeled = df_test[["id", "text"]].copy()
    df_test_solution = df_test[["id", "model"]].copy()

    # 10. Save Processed Artifacts
    print("Saving finalized datasets and metadata...")
    df_clean_final.to_parquet(out_dir / "clean_corpus_long.parquet", index=False)
    df_clean_final.to_csv(out_dir / "clean_corpus_long.csv", index=False)

    df_train_final.to_parquet(out_dir / "train_folds.parquet", index=False)
    df_train_final.to_csv(out_dir / "train_folds.csv", index=False)

    df_test_final.to_parquet(out_dir / "test_holdout.parquet", index=False)
    df_test_final.to_csv(out_dir / "test_holdout.csv", index=False)

    df_test_unlabeled.to_csv(out_dir / "test_unlabeled.csv", index=False)
    df_test_solution.to_csv(out_dir / "test_solution.csv", index=False)

    # 11. Compile Preprocessing & Feature Extraction Metadata
    extractor = RichStylometricExtractor()
    extractor.fit(["Dummy text for feature name initialization."])
    feature_names = extractor.get_feature_names_out().tolist()

    fold_distribution = {}
    for f in range(5):
        sub = df_train_final[df_train_final["fold"] == f]
        fold_distribution[f"fold_{f}"] = {
            "rows": len(sub),
            "prompts": int(sub["prompt_id"].nunique()),
            "class_counts": {k: int(v) for k, v in sub["model"].value_counts().items()}
        }

    test_distribution = {
        "rows": len(df_test_final),
        "prompts": int(df_test_final["prompt_id"].nunique()),
        "class_counts": {k: int(v) for k, v in df_test_final["model"].value_counts().items()}
    }

    audit_summary = {
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "initial_audit": initial_audit,
        "clean_audit": clean_audit,
        "split_summary": {
            "total_cleaned_rows": len(df_clean_final),
            "training_rows": len(df_train_final),
            "test_holdout_rows": len(df_test_final),
            "train_folds": fold_distribution,
            "test_distribution": test_distribution,
        },
        "file_checksums": {
            "clean_corpus_long.parquet": compute_sha256(out_dir / "clean_corpus_long.parquet"),
            "train_folds.parquet": compute_sha256(out_dir / "train_folds.parquet"),
            "test_holdout.parquet": compute_sha256(out_dir / "test_holdout.parquet"),
            "test_unlabeled.csv": compute_sha256(out_dir / "test_unlabeled.csv"),
            "test_solution.csv": compute_sha256(out_dir / "test_solution.csv")
        }
    }

    (out_dir / "data_audit_report.json").write_text(json.dumps(audit_summary, indent=2))

    preprocessing_meta = {
        "feature_engineering": {
            "multiview_architecture": "5-branch FeatureUnion",
            "branches": [
                {"name": "char", "type": "TfidfVectorizer", "analyzer": "char", "ngram_range": [1, 5], "max_features": 180000, "sublinear_tf": True},
                {"name": "word", "type": "TfidfVectorizer", "token_pattern": r"\S+", "ngram_range": [1, 2], "max_features": 150000, "sublinear_tf": True},
                {"name": "boundary", "type": "TfidfVectorizer", "analyzer": "char_wb", "ngram_range": [3, 5], "max_features": 100000, "sublinear_tf": True},
                {"name": "punct", "type": "TfidfVectorizer", "token_pattern": r"(?u)\b\w+\b|[^\w\s]", "ngram_range": [1, 2], "max_features": 100000, "sublinear_tf": True},
                {"name": "style", "type": "RichStylometricExtractor + StandardScaler", "features_count": len(feature_names), "weight": 0.2}
            ],
            "stylometric_feature_names": feature_names
        },
        "target_labels": LABELS,
        "num_classes": len(LABELS),
        "evaluation_metric": "Balanced Accuracy (macro-average recall)",
        "random_baseline": 1.0 / len(LABELS)
    }

    (out_dir / "preprocessing_metadata.json").write_text(json.dumps(preprocessing_meta, indent=2))

    print("\n================ DATA PIPELINE COMPLETED SUCCESSFULLY ================")
    print(f"Total clean passages: {len(df_clean_final):,}")
    print(f"Training passages:   {len(df_train_final):,} across 5 Stratified Group Folds")
    print(f"Holdout Test passages: {len(df_test_final):,} (strict zero-leakage prompt isolation)")
    print(f"Audit report saved:  {out_dir / 'data_audit_report.json'}")
    print(f"Preprocessing meta:  {out_dir / 'preprocessing_metadata.json'}")


if __name__ == "__main__":
    main()

