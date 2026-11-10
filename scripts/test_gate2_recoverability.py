import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering

def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("GATE 2: PROMPT CLUSTERING RECOVERABILITY AUDIT (BGE EMBEDDINGS)")
    print("=" * 70)
    print("Testing whether test passages can be grouped into prompts without labels...\n")

    # 1. Load clean competition-like prompt slice
    df = pd.read_parquet("data/processed/train_folds.parquet")
    g = df.groupby("prompt_id")["model"].agg(["count", "nunique"])
    clean_pids = g[(g["count"] == g["nunique"]) & (g["count"] >= 3)].index
    df_clean = df[df["prompt_id"].isin(clean_pids)].copy()

    # Take a balanced slice of ~1,500 passages across ~200 prompts for fast evaluation
    sampled_pids = np.random.RandomState(42).choice(clean_pids, size=min(180, len(clean_pids)), replace=False)
    df_eval = df_clean[df_clean["prompt_id"].isin(sampled_pids)].copy()

    print(f"Dataset Slice: {len(df_eval)} passages across {df_eval['prompt_id'].nunique()} unique prompts.")
    print(f"Passage counts per prompt: min={df_eval.groupby('prompt_id').size().min()}, median={df_eval.groupby('prompt_id').size().median()}, max={df_eval.groupby('prompt_id').size().max()}\n")

    texts = df_eval["text"].fillna("").tolist()
    prompt_ids = df_eval["prompt_id"].to_numpy()
    pid_codes = pd.factorize(prompt_ids)[0]
    n_prompts = len(np.unique(prompt_ids))

    # 2. Embed passages using BGE
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"Loading {model_name} (133 MB) to test BGE cosine geometry...", flush=True)
    t_emb_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Batch embedding on CPU
    batch_size = 64
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            out = model(**inputs)
            # BGE uses L2-normalized [CLS] token representation
            cls_rep = out.last_hidden_state[:, 0]
            normed = torch.nn.functional.normalize(cls_rep, p=2, dim=-1)
            all_embeddings.append(normed.cpu().numpy())
            if (i // batch_size) % 5 == 0:
                print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} passages...", flush=True)

    E = np.vstack(all_embeddings)
    t_emb = time.perf_counter() - t_emb_start
    print(f"BGE Embeddings: shape {E.shape} computed in {t_emb:.2f}s ({len(texts)/t_emb:.1f} passages/s)\n")

    # 3. Pairwise Cosine Similarity
    S = E @ E.T
    np.fill_diagonal(S, -1.0) # exclude self

    # Metric 1: Top-K Nearest-Neighbor Same-Prompt Purity
    # For each passage, evaluate top (prompt_size - 1) neighbors
    purities = []
    for i, p_code in enumerate(pid_codes):
        p_size = np.sum(pid_codes == p_code)
        k = max(1, p_size - 1)
        top_k_idx = np.argpartition(-S[i], k)[:k]
        purity_i = np.mean(pid_codes[top_k_idx] == p_code)
        purities.append(purity_i)
    mean_purity = np.mean(purities) * 100
    median_purity = np.median(purities) * 100
    purity_90_pct = np.mean(np.array(purities) >= 0.90) * 100
    purity_100_pct = np.mean(np.array(purities) == 1.0) * 100

    print("--- METRIC 1: NEAREST-NEIGHBOR SAME-PROMPT PURITY ---")
    print(f"Mean Top-K Same-Prompt Purity:      {mean_purity:.2f}%")
    print(f"Median Top-K Same-Prompt Purity:    {median_purity:.2f}%")
    print(f"Passages with >=90% Sibling Purity: {purity_90_pct:.1f}%")
    print(f"Passages with 100% Sibling Purity:  {purity_100_pct:.1f}%\n")

    # Metric 2: Cosine Similarity Contrast (Same Prompt vs Different Prompts)
    same_prompt_mask = (pid_codes[:, None] == pid_codes[None, :])
    np.fill_diagonal(same_prompt_mask, False)
    diff_prompt_mask = ~same_prompt_mask
    np.fill_diagonal(diff_prompt_mask, False)

    same_sims = S[same_prompt_mask]
    diff_sims = S[diff_prompt_mask]

    print("--- METRIC 2: COSINE SIMILARITY DISCRIMINABILITY ---")
    print(f"Same-Prompt Cosine Sim:     mean={same_sims.mean():.4f} +/- {same_sims.std():.4f} (5%={np.percentile(same_sims, 5):.4f}, 50%={np.median(same_sims):.4f})")
    print(f"Different-Prompt Cosine Sim: mean={diff_sims.mean():.4f} +/- {diff_sims.std():.4f} (50%={np.median(diff_sims):.4f}, 95%={np.percentile(diff_sims, 95):.4f})")
    
    # Fast AUC calculation on subsample of pairs to keep runtime tiny
    rng = np.random.RandomState(42)
    s_sub = rng.choice(same_sims, size=min(10000, len(same_sims)), replace=False)
    d_sub = rng.choice(diff_sims, size=min(50000, len(diff_sims)), replace=False)
    y_pair = np.concatenate([np.ones(len(s_sub)), np.zeros(len(d_sub))])
    score_pair = np.concatenate([s_sub, d_sub])
    pair_auc = roc_auc_score(y_pair, score_pair)
    print(f"Pairwise Prompt Discrimination AUC: {pair_auc:.4f} (1.0 = perfect separation)\n")

    # Metric 3: Agglomerative Clustering Across Distance Thresholds
    print("--- METRIC 3: AGGLOMERATIVE CLUSTERING RECOVERY ---")
    best_ari = -1.0
    best_thresh = None
    dist_matrix = 1.0 - (E @ E.T)
    np.fill_diagonal(dist_matrix, 0.0)

    for thresh in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        cl = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thresh,
            metric="precomputed",
            linkage="average"
        ).fit_predict(dist_matrix)
        ari = adjusted_rand_score(pid_codes, cl)
        nmi = normalized_mutual_info_score(pid_codes, cl)
        n_clusters = len(np.unique(cl))
        sizes = pd.Series(cl).value_counts()
        print(f"  Threshold d<={thresh:.2f} (cos>={1-thresh:.2f}): Clusters={n_clusters:3d} (true={n_prompts}) | ARI={ari:.4f} | NMI={nmi:.4f} | Median Size={sizes.median():.0f}")
        if ari > best_ari:
            best_ari = ari
            best_thresh = thresh

    print("\n" + "=" * 70)
    print("GATE 2 VERDICT & CLARIFICATION")
    print("=" * 70)
    print(f"Best Recovered Cluster ARI:         {best_ari:.4f} (at cos >= {1-best_thresh:.2f})")
    print(f"Top-K Sibling Purity:              {mean_purity:.2f}%")
    print(f"Pairwise Discrimination AUC:        {pair_auc:.4f}")
    print("=" * 70)

    if best_ari >= 0.80 and mean_purity >= 90.0:
        print("VERDICT: GATE 2 PASSED [OK].")
        print("BGE embeddings accurately recover prompt groups without labels (Purity >= 90%, ARI >= 0.80).")
        print("Sibling assignment CAN be safely deployed on high-confidence recovered clusters!")
    else:
        print(f"VERDICT: GATE 2 FAILED [STOP].")
        print("Cluster recovery is too noisy. Sibling decoding would suffer negative transfer.")

if __name__ == "__main__":
    main()

