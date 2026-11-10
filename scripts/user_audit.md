<USER_REQUEST>
# Audit: Quad-Model Ensemble on Kaggle T4×2

I haven't run this. Where I give runtime or memory numbers, they are estimates that scale with your training-set size **N** and average text length, which the script never measures. Findings are ordered by severity.

## 1. Likely run-killers or silently wrong results

**1.1 The time budget is almost certainly infeasible, and the guard doesn't protect you.**
- Rough cost of one sample (forward, backward and checkpoint recompute) at 352 tokens:
  - Qwen 7B costs about 2×10¹³ FLOPs. With `device_map="auto"` only one GPU is busy at a time, so expect about 1 sample/s on T4s.
  - DeBERTa-v3-large runs at roughly 8–12 samples/s on T4 with checkpointing.
- At N=30k, Qwen takes about 8 h per epoch and DeBERTa about 3 h per fold. Five folds of everything cannot fit in 9 h unless N is only a few thousand.
- The time guard is checked only at the start of a fold (`fold > 0`), so a single fold can run past the limit with nothing saved.
- The first `submission.parquet` is written after all four models finish fold 0, which could be hours in.
- Fix:
  - Run the cheap CPU model first and write a submission immediately.
  - Time a 200-sample step benchmark at start-up and compute a feasible epochs/folds plan.
  - Add intra-fold wall-clock checks.
  - Consider Qwen on a single holdout split, or a 1.5B/3B model, instead of 5×7B.

**1.2 A failed fold poisons the OOF arrays.**
- Each model is wrapped in `except Exception`, which prints only `e` (no traceback) and leaves `oof_*[va_idx]` as zeros.
- Those rows still pass `eval_mask`, because that mask is built only from `oof_svm`.
- Zero logits give `argmax = 0` (gpt-5.5) and constant ranks, which corrupts the per-model scores and the blend-weight search.
- Fix: initialize the OOF arrays with `np.nan` and mask per model, call `traceback.print_exc()`, and run a 1-epoch, 300-sample smoke test of every model before the real loop. A systematic bug otherwise burns hours.

**1.3 Qwen stream has several independent ways to fail or silently vanish.**
- Only the Qwen load uses `dtype=`, with no `torch_dtype` fallback (the other two have one). On a transformers version older than about 4.56 this raises, and the outer `except` skips Qwen on every fold.
- Qwen2.5 in fp16 autocast has well-known overflow/NaN issues. T4 has no usable bf16, and this loop has no NaN guard (DeBERTa has one, BGE and Qwen don't).
- A NaN loss makes `GradScaler` shrink its scale indefinitely while nothing trains.
- GradScaler also requires fp32 trainable params. Recent peft upcasts adapters automatically, but verify `lora_params[0].dtype == torch.float32`. Older peft raises "Attempting to unscale FP16 gradients".
- The OOM handler leaves `out/pooled/logits/loss` and the traceback alive, so `empty_cache()` can't free much and the retry likely OOMs again.
- After a batch-size halving, the optimizer-step count no longer matches `total_steps`, so the cosine schedule is cut short and the effective batch size silently changes.
- Fix: `del` the locals and `gc.collect()` before retrying, and schedule on actual steps or use a fixed micro-batch.

**1.4 The CPU branch will likely exceed the claimed "~6 GB".**
- `TfidfVectorizer(analyzer="char", ngram_range=(1,6))` builds the full vocabulary dict before `max_features` prunes it. That is tens of millions of Python dict entries, easily 10+ GB for the char branch alone.
- The sparse matrix has on the order of 10⁴ nnz per document. liblinear then makes its own float64 copy, so peak RAM can approach the 30 GB limit, alongside pinned tensors and CPU state-dict copies.
- `LinearSVC(dual=True, max_iter=20000)` over 1.2M features and 18 one-vs-rest problems can run for hours without converging.
- It is also refit and re-transformed from scratch for each of 5 folds, plus val and test.
- Fix:
  - Fit the vectorizers once on train+test text (unsupervised) or use `HashingVectorizer` + `TfidfTransformer`.
  - Lower `max_iter` to about 2000.
  - Cut the n-gram range to (2,5).
  - Run it in a separate process.

**1.5 Gating and Sinkhorn operate on the wrong scale.**
- `apply_hierarchical_family_gating` takes a softmax of rank sums. Adjacent ranks differ by Σw, so with Σw≈5 the softmax is nearly one-hot. Family probabilities become 0/1 and `log(1e-12)·0.35 ≈ −9.7`, so "soft" gating becomes hard gating.
- The interim checkpoint uses Σw=1.0, which is far softer, so interim and final behave differently.
- The grid search over weights also tunes the gating strength by accident, because absolute scale and relative ratio are confounded.
- Family mass also favors multi-member families (OpenAI and Anthropic have 5 each) over singletons (qwen, kimi), which is not a neutral prior.
- Sinkhorn with `temperature=1.5` on that scale gives a nearly one-hot matrix that converges very slowly.
  - The `tol=1e-3` check is on the absolute change of `c`, which is meaningless when `c` spans orders of magnitude.
  - Balancing is barely applied after 50 iterations.
- Sinkhorn is never validated on OOF. It also forces uniform class counts, which is wrong if the hidden test mirrors the (evidently imbalanced, ~70× per your own comment) train distribution.
- Fix: compute per-model calibrated log-probs (temperature-scale on OOF), average them, apply gating and Sinkhorn to real probabilities, and test the Sinkhorn step on OOF with a resampled uniform subset.

## 2. ML and optimization issues

- **LLRD of 0.85 over 24 layers is too steep.** Layer 12 gets about 17% of the base LR and the embeddings about 2%, so the lower half barely trains. Authorship attribution leans heavily on low-level lexical and formatting signals. Use 0.90–0.95. DeBERTa's `encoder.rel_embeddings`/`LayerNorm` fall into the leftover group at full LR, inconsistent with the layers.
- **Tokenization destroys the strongest LLM-fingerprint signals.**
  - BGE-large-en-v1.5 uses an uncased WordPiece vocab, so casing is lost.
  - DeBERTa's SentencePiece collapses newlines.
  - Neither model sees `\n\n`, list structure, or whitespace artifacts.
  - Fix: map `\n` to a visible token (e.g. `[NL]`) before tokenizing.
  - Right-only truncation also drops the ending, where models differ most ("In summary…"). Use head+tail truncation.
- **`clean_text` removes discriminative artifacts** (`\r\n`, NBSP, zero-width chars, BOM), which can themselves be model fingerprints. Keep the raw text for the char-level TF-IDF branch. The fast path also skips the control-character regex, so cleaning is inconsistent.
- **Everything is padded to 352.** `padding=True` on the full set means short texts waste compute. Length-bucketed batches with per-batch trimming typically give 1.4–2× throughput.
- **Gradient checkpointing is probably unnecessary for DeBERTa and BGE.** They hold about 7–8 GB for weights, grads and Adam states, leaving room for BS 10–12 at 352 tokens on 15 GB. Checkpointing costs about 25–30% extra compute. Test it off, or keep it on and raise the batch size.
- **OOF is optimistically biased.** The best epoch is chosen on the same validation fold that produces the OOF logits. Blend weights are then grid-searched on those OOF scores (1,080 combinations) without Sinkhorn, so the reported OOF isn't the submission pipeline.
- **Borda ranks throw away confidence.** Summed ranks of low-ranked classes add noise. A logistic-regression stacker on the concatenated calibrated logits (72 features, nested CV) usually beats rank-sum plus grid weights.
- **The class prior is corrected three times**: weighted CE, SVM `class_weight="balanced"`, and Sinkhorn. The docstring claim "caps disparity at ~4.5×" is wrong: after clipping to [0.4, 4.0] the ratio is up to 10×.
- **Qwen LoRA LR of 3.5e-5 is low** for r=64/α=128 (typical 1e-4 to 2e-4) over only 3 epochs, so it is probably underfit. The comment says 80M trainable parameters, but the actual count is about 160M.
- **No seeds** are set for torch or numpy (`np.random.permutation` is unseeded), so runs aren't reproducible.
- **`train_bge_fold` is a near-duplicate of the DeBERTa function but drifted**: it has no NaN/FP32 fallback and clips at 1.0 instead of 0.5. Merge them into one function.
- **DeBERTa NaN path**: `continue` skips the update but still advances `micro_step`, which misaligns accumulation. It also permanently drops to FP32, which on T4 is extremely slow and may OOM.

## 3. Kaggle runtime, concurrency and hardware

- **No true deadlock risk.** `HeartbeatTimer` is a daemon thread and exits cleanly, and the tokenizer/offline env vars are set before the imports that read them. The real "hang" candidates are the LinearSVC non-convergence above, slow Sinkhorn, and the `rglob` over `/kaggle/input`.
- **"Max hardware utilization" is not what happens.** Everything runs strictly sequentially: GPU1 is idle during DeBERTa, GPU0 is idle during BGE, the CPUs are idle during all GPU work, and Qwen's naive pipeline keeps one GPU idle at a time.
  - DeBERTa (cuda:0) and BGE (cuda:1) are independent and could run concurrently, ideally as separate processes.
  - The CPU model could run in a third process at the same time.
- **`torch.cuda.empty_cache()` only clears the current device.** The `for gi in range(n_gpus)` loops in `main` synchronize both GPUs but empty only one, and `set_device(cuda:1)` leaks the "current device" into later code. Use `with torch.cuda.device(i): torch.cuda.empty_cache()`.
- **Models are reloaded from disk every fold.** That is 15 GB for Qwen alone, 5 times.
- **`pin_memory()` on the Qwen tensors does nothing useful**, because advanced indexing (`tr_input_ids[idx]`) returns a non-pinned copy, so `non_blocking` isn't async. Indexing CUDA tensors with a NumPy `idx` also forces a host round-trip. Use a torch index tensor on the device.
- **Per-step `.item()` and `isnan` calls force syncs.** Accumulate on the GPU and log every N steps.
- **`expandable_segments:True`** can warn or fall back on some Kaggle driver versions. It's harmless but worth checking in the log.

## 4. Smaller correctness and cosmetic items

- `RichStyleFeatures` actually produces about 132 features (15 + 54 punctuation + 41 function words + 7 positional + 15 paragraph/density), not 115. Verify with `.shape`.
- The `['']` character class in `CONTRACTION_RE`/`WORD_RE` is a duplicated straight apostrophe, so it likely lost the curly `’` and misses curly-quote contractions.
- The `lower.count(fw)` loop for 49 function words is O(49·nw) per document; use a `Counter`.
- `DSU.find` is recursive and could hit Python's recursion limit on long chains.
- `compute_passage_ranks` uses a per-row Python loop; `scipy.stats.rankdata(m, axis=1)` is vectorized.
- `best_state = {k: v.cpu().clone()}` copies twice, since `.cpu()` already copies.
- The Qwen mean-pooling code with left padding is fine, but position ids shift under left padding (RoPE is relative, so it's minor).
- Unused imports: `concurrent.futures`, `CosineAnnealingLR`.
- `len(sub) != len(test_ids)` in the submission validator is a tautology.

## Suggested restructure

1. Run a smoke test of every model on a tiny subset, plus a throughput benchmark that sets epochs and folds from the time budget.
2. Fit the CPU model first (shared vectorizers) and write a submission immediately.
3. Run DeBERTa and BGE concurrently on the two GPUs, with newline tokens, dynamic padding, LLRD of about 0.92, and consistent NaN handling.
4. Make Qwen optional: a single split or a smaller model, with an fp16 NaN guard.
5. Calibrate, then stack or average log-probs, apply gating and Sinkhorn on probabilities, and validate both on OOF.
6. Add per-model and intra-fold time checks that save a submission after each stage.

If you tell me N, the average text length, and which of the Qwen/DeBERTa/BGE checkpoints are actually attached, I can turn the budget estimates into concrete epoch/fold numbers and draft the refactored training loop.


Review this negatively against the codebase and audit each issue thoroughly. Verdict for all of them properly.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-30T23:46:10+05:30.

The user's current state is as follows:
Active Document: d:\NameThatLLM\notebooks\kaggle_tri_ensemble.py (LANGUAGE_UNSPECIFIED)
Cursor is on line: 1
</ADDITIONAL_METADATA>