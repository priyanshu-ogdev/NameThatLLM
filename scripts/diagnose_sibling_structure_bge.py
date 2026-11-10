"""
Diagnostic: Sibling Structure Verification via Frozen BGE-large-en-v1.5.
Implements the User's Audit §4 Specification:
1. Embeds passages with frozen BGE-large-en-v1.5 (cuda or cpu).
2. Computes top-17 nearest-neighbors prompt purity: fraction of 17 nearest neighbors sharing true prompt_id.
3. Computes Agglomerative Clustering ARI and NMI against true prompt_ids.
4. Performs label-free cluster-size check on test passages: tests if cluster sizes concentrate around 18.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

@torch.no_grad()
def bge_embed(texts, model_path, device="cuda:0", bs=64):
    """Generates L2-normalized [CLS] embeddings from frozen BGE-large."""
    if not torch.cuda.is_available():
        device = "cpu"
    print(f"Loading BGE from {model_path} on {device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    m = AutoModel.from_pretrained(
        model_path, 
        local_files_only=True, 
        torch_dtype=torch.float16 if "cuda" in device else torch.float32
    ).to(device).eval()
    
    embeddings = []
    for i in range(0, len(texts), bs):
        batch_texts = texts[i:i+bs]
        enc = tok(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = m(**enc)
        # BGE uses normalized [CLS] token representation
        cls_rep = out.last_hidden_state[:, 0].float()
        normed = torch.nn.functional.normalize(cls_rep, dim=-1).cpu()
        embeddings.append(normed)
        if (i // bs) % 10 == 0:
            print(f"  Embedded {min(i+bs, len(texts))}/{len(texts)} passages...", flush=True)
            
    return torch.cat(embeddings).numpy()

def evaluate_sibling_structure(train_df, test_df, bge_path):
    print("=" * 75, flush=True)
    print("BGE SEMANTIC SIBLING STRUCTURE DIAGNOSTIC (USER §4 PROTOCOL)", flush=True)
    print("=" * 75, flush=True)
    
    # 1. Embed Train
    print("\n[Step 1/3] Embedding Train Passages...", flush=True)
    train_texts = train_df["text"].fillna("").tolist()
    E_tr = bge_embed(train_texts, bge_path)
    print(f"Train embeddings shape: {E_tr.shape}")
    
    # 2. Top-17 Nearest-Neighbor Purity
    print("\n[Step 2/3] Evaluating Ground-Truth Prompt Purity...", flush=True)
    pid = train_df["prompt_id"].factorize()[0]
    n_prompts = train_df["prompt_id"].nunique()
    
    S = E_tr @ E_tr.T
    np.fill_diagonal(S, -1.0)
    # top 17 neighbors
    nn_idx = np.argpartition(-S, 16, axis=1)[:, :17]
    purity = (pid[nn_idx] == pid[:, None]).mean()
    print(f"  Top-17 Nearest-Neighbor Same-Prompt Purity: {purity:.4f} ({purity*100:.2f}%)")
    
    # Agglomerative Clustering
    cl = AgglomerativeClustering(n_clusters=n_prompts, metric="cosine", linkage="average").fit_predict(E_tr)
    ari = adjusted_rand_score(pid, cl)
    nmi = normalized_mutual_info_score(pid, cl)
    cluster_sizes = np.bincount(cl)
    pctls = np.percentile(cluster_sizes, [5, 25, 50, 75, 95])
    print(f"  Agglomerative Clustering: ARI = {ari:.4f} | NMI = {nmi:.4f}")
    print(f"  Cluster Count: {len(set(cl))} | Size Percentiles [5, 25, 50, 75, 95]: {pctls}")
    
    # Decision Gate
    structure_viable = (purity >= 0.85 and ari >= 0.75)
    print(f"  => Sibling Structure Viability Gate: {'PASSED [OK]' if structure_viable else 'FAILED [STOP]'}")
    
    # 3. Label-Free Check on Test Set
    print("\n[Step 3/3] Performing Label-Free Check on Test Set...", flush=True)
    test_texts = test_df["text"].fillna("").tolist()
    if len(test_texts) >= 18:
        E_te = bge_embed(test_texts, bge_path)
        k_test = max(1, round(len(E_te) / 18))
        ct = AgglomerativeClustering(n_clusters=k_test, metric="cosine", linkage="average").fit_predict(E_te)
        ct_sizes = np.bincount(ct)
        ct_pctls = np.percentile(ct_sizes, [5, 25, 50, 75, 95])
        print(f"  Test Passages: {len(test_texts)} | Target Clusters (N/18): {k_test}")
        print(f"  Test Cluster-Size Percentiles [5, 25, 50, 75, 95]: {ct_pctls}")
        test_viable = (ct_pctls[1] >= 14 and ct_pctls[3] <= 22)
        print(f"  => Test Family Concentration Gate: {'PASSED (Concentrated around 18)' if test_viable else 'SCATTERED / ISOLATED'}")
    else:
        print(f"  Test set has only {len(test_texts)} rows (smoke test slice). Sibling clustering requires N >= 18.")

if __name__ == "__main__":
    bge_dir = os.environ.get("BGE_DIR", "/tmp/model_cache_bge")
    # Fallback to local paths if needed
    if not os.path.exists(bge_dir):
        candidates = [
            "/kaggle/input/datasets/wuwenmin/bge-large-en-v1-5",
            "models/bge-large-en-v1-5"
        ]
        for c in candidates:
            if os.path.exists(c):
                bge_dir = c
                break
    print(f"BGE Directory: {bge_dir}")

