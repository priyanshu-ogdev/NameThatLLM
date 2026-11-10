<USER_REQUEST>
# Audit of the "negative audit" report

I haven't run any of this. I'm reading the report against your earlier logs.

## Verdict

**"Don't ship CPU v2 as-is" is a reasonable default, but the evidence behind it is much weaker than the report claims.**
- Most of the rejected experiments aren't the experiments I proposed.
- Several explanations contradict the design.
- The harness doesn't reproduce your measured 35.56%.
- The report quietly closes the highest-value lever, sibling structure, on a degenerate test.

Some of this is my fault. Where I got things wrong, I say so in §3 and §7.

## 1. The harness doesn't match reality

| Evidence | Value |
|---|---|
| Your measured grouped 5-fold CV on the full train set (Kaggle log) | **35.56%** (folds 33.3–38.0) |
| Report baseline, rows: 167-dim, PromptDF≥5, ExtraTrees | **63.27%** |
| Report baseline, rows: masking, PromptDF≥3 | **35.94%** |
| N used | 2,023 (the full train set is 2,699) |

- **Two baselines in one table.** A 63% harness and a 36% harness are mixed, so the PromptDF≥3 (−1.7) versus ≥5 (+3.1) comparison isn't a threshold sweep. The two rows come from different setups.
- **63% is about 1.8× your measured score.**
  - It is also about 1.7× the 37.2% CPU baseline attributed to the public notebook, which I can't verify.
  - ExtraTrees on stylometrics alone at 61.2% would nearly match the full pipeline. That contradicts the 35.56% result on the same data.
  - Until the harness reproduces ≈35.6% on all 2,699 rows with identical folds, I wouldn't trust any row measured against 63%.
- **Fold 1 versus "mean".** The text says "in Fold 1, 63.27 → 60.00", but the table calls those means and reports "0/5 wins". Either the table is mislabeled or only one fold was run.
- **Smaller sloppiness.** "Mean 18.0, std 0.0 models per prompt" can't hold if claude-fable-5 has 149 rows, because one prompt must have 17 models.

## 2. Row-by-row verdicts

| Item | Report | Mine | Why |
|---|---|---|---|
| 167-dim dense features | Rejected (−3.27) | **Invalid test** | I proposed a separate scaled dense stream (LogReg/LightGBM on V). The report appended raw features to the sparse union. The explanation is "unscaled", which is an implementation flaw, not evidence against the features. |
| Topic masking | Collapse to 11.11% | **Unresolved, probably a bug or bad threshold** | 11.11% is suspiciously round, so check the predicted-class histogram. I said to add masking as an extra branch, and they seem to have replaced the text. The 40%-of-prompts threshold was my aggressive default. The explanation contradicts the design: "furthermore" appears in many prompts and so survives the keep-list by construction. |
| PromptDF≥3 | −1.7 | **Not comparable** | Different baseline (see §1). |
| PromptDF≥5 | +3.09, 3/5, "rejected" | **Inconclusive, and rejected on the wrong grounds** | "Breaks the 30-second architecture" is a convenience argument against a +3 point gain. The cost claim is also doubtful (§5). |
| ExtraTrees | Rejected | **Wrong criterion** | A blend member is judged by blend delta, not standalone parity. LightGBM wasn't tested. "Windows" is irrelevant on Kaggle, and ExtraTrees `n_jobs` uses threads. "Blends badly" has no number attached. |
| Pseudo-prompt clustering | ARI −0.04, NMI 0.0000 | **Degenerate test** | NMI of exactly 0 means one side of the comparison is a constant partition. Check how many clusters it produced. |
| Qwen-0.5B LM stats | "Fatal crash" | **Not an accuracy result** | A 0.5B model is a tiny attachable Kaggle model. Alternatively, take the stats from the Qwen-7B you already load. |

## 3. What the data check establishes, and a correction

Confirmed: 150 prompts, 150 rows per model (149 for claude-fable-5), at most one answer per model per prompt. Train is balanced, so class weights, `gamma` and "rare-class starvation" are moot. Sinkhorn's uniform assumption is plausible, if the test set is similarly balanced (unconfirmed).

**Correction to my earlier reasoning.** I said topic n-grams cause *leakage*. With every prompt answered once by every model, topic is nearly independent of the label in train. Topic words therefore add **variance** (p ≫ n, about 850k features on about 2.2k rows per fold), not systematic bias. This predicts:
- Masking shouldn't be a big winner.
- Stronger regularization and tuning should matter more than I implied.
- `C=1.2` and `style_weight=0.2` were never tuned on grouped CV. That is probably the cheapest unexplored gain.

## 4. The sibling-structure rejection isn't valid

"No `prompt_id` in test, so within-prompt methods are impossible" rests on one TF-IDF clustering run with a degenerate result. Eighteen answers to the same prompt should be very close in a semantic embedding space. Retest properly with frozen BGE, which is already attached:

```python
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

@torch.no_grad()
def bge_embed(texts, path, dev="cuda:0", bs=64):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    m = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16).to(dev).eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        out.append(torch.nn.functional.normalize(m(**enc).last_hidden_state[:, 0].float(), dim=-1).cpu())
    return torch.cat(out).numpy()

pid = train_df["prompt_id"].factorize()[0]
E = bge_embed(train_df["text"].tolist(), BGE_DIR)
S = E @ E.T; np.fill_diagonal(S, -1)
nn_idx = np.argpartition(-S, 16, axis=1)[:, :17]
print("top-17 same-prompt purity:", (pid[nn_idx] == pid[:, None]).mean())

cl = AgglomerativeClustering(n_clusters=150, metric="cosine", linkage="average").fit_predict(E)
print("ARI", adjusted_rand_score(pid, cl), "NMI", normalized_mutual_info_score(pid, cl),
      "n_clusters", len(set(cl)), "size pctl", np.percentile(np.bincount(cl), [5, 50, 95]))

# Does the TEST set have the same structure? Label-free check.
Et = bge_embed(test_df["text"].tolist(), BGE_DIR)
ct = AgglomerativeClustering(n_clusters=round(len(Et)/18), metric="cosine", linkage="average").fit_predict(Et)
print("test cluster-size pctl:", np.percentile(np.bincount(ct), [5, 25, 50, 75, 95]))
```

**Reading it.**
- If purity is above about 0.9, ARI above about 0.8, and test cluster sizes concentrate near 18, the structure is real and recoverable. If purity is low or test sizes are scattered, drop the idea.
- If it passes, there are two upgrades, in order:
  1. **Neighborhood centering.** Subtract the mean of each row's 17 nearest neighbors from its dense features and from the neural log-probs at inference. Apply the same label-free procedure in training, with neighbors found *without* `prompt_id`, so recovery errors are baked in consistently.
  2. **Assignment decoding.** Run Hungarian assignment inside each recovered cluster, as in my earlier `decode_by_prompt` sketch. Validate on OOF using recovered clusters, not true ids, with the pool at the full train set (150 prompts, similar in difficulty to a 222-prompt test).
- Whether any of this helps is an empirical question. If it works, it could be larger than any CPU-model change, and if it doesn't, it costs minutes to find out.

## 5. Re-testing the rejected items properly

All in one harness, on the full 2,699 rows, 3 seeds × 5 grouped folds, paired per fold.

1. **Reproduce the baseline** at about 35.6% ±2. If it doesn't, stop and fix the harness first.
2. **PromptDF sweep** over `min_prompts` ∈ {2, 3, 5, 8, 12}, fold-isolated, with no column copies:

```python
import scipy.sparse as sp, pandas as pd
def prompt_df_mask(Xtr, gtr, min_prompts):
    B = (Xtr > 0).astype(np.float32).tocsr()
    codes = pd.factorize(gtr)[0]
    G = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))))
    pdf = np.asarray(((G @ B) > 0).sum(axis=0)).ravel()
    return (pdf >= min_prompts).astype(np.float32)
# apply: Xtr_f = Xtr @ sp.diags(mask); Xva_f = Xva @ sp.diags(mask)   # time it
```

   The report says this costs about 4.5 minutes and causes memory spikes. On a roughly 2.7k × 850k CSR (about 120 MB by your own earlier figure), it should take about a second. Measure it before accepting the claim.
3. **Regularization first**: tune `C`, `style_weight` and per-block weights, and try LogisticRegression and NB-SVM. This is cheap and addresses the variance diagnosis directly.
4. **Dense stream**: scale V inside each fold, then compare LogReg/LightGBM/ExtraTrees on V, standalone and by blend delta.
5. **Masking**: sweep `min_frac` ∈ {0.05, 0.1, 0.2}, as an extra branch, and log the predicted-class histogram.
6. **LM statistics**: attach Qwen-0.5B as a model input, or use the 7B's forward pass. This is the only item that was infeasible rather than negative.

## 6. Further upgrades, ranked

1. **Harness reconciliation (§1).** Nothing else is decidable without it.
2. **Sibling structure test (§4).** It is cheap and potentially the biggest lever.
3. **Regularization and block-weight tuning for the CPU stream.** The cheapest likely gain.
4. **Separate processes instead of threads for dual-GPU training.** The report's "DeBERTa on GPU 0, BGE on GPU 1" in one process shares the GIL, the global RNG, `set_device`, and the per-device cache clearing. Isolate them:

```python
procs = [subprocess.Popen([sys.executable, "train_fold.py", "--model", m, "--fold", str(f)],
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": g})
         for m, g in (("deberta", "0"), ("bge", "1"))]
for p in procs: p.wait(timeout=...)   # worker saves val/test logits to .npz
```

   This also frees VRAM fully per fold, isolates OOMs, and gives hard time limits.
5. **Frozen mid-layer Qwen and BGE embeddings with neighborhood centering and a LogReg head.** Cheap, diverse, and a fraction of LoRA cost.
6. **A real single-fold neural run** (not 16 samples) for accuracy and throughput. The "3.42 h", "4.99 h" and "5.95 h" runtime figures keep changing, and none is measured.
7. **A grouped-CV LogReg stacker** over per-stream OOF log-probs, with NLL-fitted temperatures. The grid-searched weights must beat it.

## 7. Revised acceptance protocol (my earlier rule was weak)

- **4-of-5 folds isn't evidence.** A one-sided sign test gives p ≈ 0.19 for 4/5 and p ≈ 0.03 for 5/5. Use 15 paired folds (3 seeds × 5), a prompt-level paired bootstrap, and require mean gain ≥ 1.5 with the CI lower bound above 0.
- **Single-model feature changes** are judged on standalone CV.
- **New ensemble streams** are judged by blend delta (stacker with versus without the stream). Standalone parity is the wrong test.
- **Gate the dry run on reality.** If the baseline CV is outside about 33–38%, or any stream is near chance, fail loudly. "11/11 passed" and "byte parity across 1,932 lines" verify consistency between two files, not accuracy.

## What I'd paste back for the next round

1. The harness output reproducing the baseline on all 2,699 rows (per-fold scores).
2. The §4 diagnostics: purity, ARI/NMI, cluster counts, and test cluster-size percentiles.
3. The PromptDF sweep table with 15 paired folds.
4. One real neural fold's validation score and throughput.


Review this. What do you think??? Analyze and audit against our reasoning.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-01T13:35:33+05:30.

The user's current state is as follows:
Active Document: d:\NameThatLLM\notebooks\kaggle_dry_run.py (LANGUAGE_UNSPECIFIED)
Cursor is on line: 1
</ADDITIONAL_METADATA>