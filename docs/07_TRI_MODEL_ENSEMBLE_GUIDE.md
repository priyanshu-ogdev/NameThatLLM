# Name That LLM: Tri-Model Ensemble Specification & Kaggle Execution Guide

## 1. Executive Architecture Overview

The **Tri-Model Ensemble** brings together three complementary modeling paradigms into a unified, leak-free pipeline designed to maximize **Balanced Accuracy** in the Kaggle **Name That LLM** competition:

```
                                  Anonymous Text Passage (40–250 words)
                                                    │
                 ┌──────────────────────────────────┼──────────────────────────────────┐
                 ▼                                  ▼                                  ▼
        ┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
        │  Multiview CPU   │               │   Frozen Qwen    │               │  DeBERTa-v3      │
        │  Stylometric SVM │               │   8B / 9B Base   │               │  Large / Base    │
        │ (Surface Signals)│               │  (Causal Syntax) │               │ (Bidirectional)  │
        ├──────────────────┤               ├──────────────────┤               ├──────────────────┤
        │ • Em-dashes (—)  │               │ • Autoregressive │               │ • Disentangled   │
        │ • Curly quotes (“│               │   next-token flow│               │   relative pos.  │
        │ • Markdown (#, `)│               │ • Tokenizer BPE  │               │ • Whole-passage  │
        │ • Char-boundary  │               │   merges & bias  │               │   grammaticality │
        │   n-grams (wb)   │               │ • Middle layers  │               │ • Clause bounds  │
        │ • Rich style     │               │   (18, 27, 36)   │               │ • Mean pooled    │
        └────────┬─────────┘               └────────┬─────────┘               └────────┬─────────┘
                 │                                  │                                  │
                 ▼                                  ▼                                  ▼
           18 Continuous                      18 Continuous                      18 Continuous
              Margins                            Margins                            Margins
                 │                                  │                                  │
                 ▼                                  ▼                                  ▼
           Within-Passage                     Within-Passage                     Within-Passage
            Ranks (1..18)                      Ranks (1..18)                      Ranks (1..18)
                 │                                  │                                  │
                 └──────────────────────────────────┼──────────────────────────────────┘
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ Zero-Parameter Rank Summation │
                                    │ Rank_Total = R_SVM +          │
                                    │              R_Qwen +         │
                                    │              R_DeBERTa        │
                                    │ (Ties broken by SVM margin)   │
                                    └───────────────┬───────────────┘
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ Argmax -> Winning Model Class │
                                    └───────────────────────────────┘
```

---

## 2. Why This Combination Dominates

| Model | Structural Role | What It Detects That Others Miss | Blindspots It Overcomes |
| :--- | :--- | :--- | :--- |
| **Qwen 8B / 9B Base** | Causal Autoregressive Generative Backbone | Causal probability flow, RLHF alignment hedging, and tokenizer BPE biases. Crucially, provides **native resonance with `qwen3.8-max`** (one of the 18 target authors) and `kimi-k3`. | Weak at holistic sentence-level bidirectional clause checking. |
| **DeBERTa-v3-large** | Bidirectional Masked MLM | Disentangled relative position attention. Captures long-range syntactic dependencies and grammatical agreement across the entire passage simultaneously. | Misses exact typographic symbols compressed by subwords. |
| **Multiview CPU SVM** | Deterministic Orthographic Anchor | Exact counts and frequencies of punctuation (em-dashes `—`, en-dashes `–`, curly quotes `“`), markdown headers (`#`), code backticks (`` ` ``), and character boundaries (`char_wb`). | Lacks contextual neural semantics. |

---

## 3. Kaggle Setup & Resource Budgeting

Submissions run inside an offline Kaggle container with **Internet strictly disabled** and a **9-hour execution limit**.

### Required Kaggle Settings:
- **Accelerator:** `GPU T4 x2` (Dual Tesla T4, 16 GB each = 32 GB total VRAM).
- **Internet:** `Off`.
- **Attached Notebook Inputs:**
  1. **Competition Data:** `Name That LLM` (contains `train.csv` and `test.csv`).
  2. **Generative Model:** `qwen-lm/qwen-3/transformers/8b-base/1` (or attached Qwen 3.5/3.8 9B weights).
  3. **Bidirectional Model:** `microsoft/deberta-v3-large` (or `deberta-v3-base`).

### Hardware & Memory Plan:
- **Qwen 8B/9B (FP16, ~16 GB):** Distributed across GPU 0 (~8 GB) and GPU 1 (~8 GB) using Hugging Face `device_map="balanced"`. Both GPUs retain ~8 GB of headroom.
- **DeBERTa-v3 (FP16, ~1.7 GB):** Allocated to GPU 1 (taking GPU 1 to ~9.7 GB / 16 GB).
- **Multiview SVM:** Runs on the 4-core Intel Xeon CPU, consuming 0 MB of GPU memory.
- **Total Runtime:** $\approx 18\text{ min}$ (train) + $\approx 25\text{ min}$ (hidden test $\approx 4,000\text{ rows}$) $\approx 45\text{ to } 50\text{ minutes}$, leaving 8+ hours of safety margin.

---

## 4. Zero-Parameter Within-Passage Rank Blending

### Why Rank Summation Beats Softmax Probability Averaging:
1. **Uncalibrated Margins:** SVM decision distances, DeBERTa classification logits, and Qwen linear margins operate on disparate, uncalibrated numerical scales.
2. **Temperature Sensitivity:** Fitting a softmax temperature parameter on cross-validation risks overfitting to the training prompts and collapsing on the unseen hidden test set.
3. **Scale Invariance:** For passage $i$, each model's 18 decision margins are transformed into ranks from $1$ (lowest) to $18$ (highest):
   $$\text{Rank}(i, c) = \text{rank}(\text{Margin}_{i, c}) \in [1, 18]$$
4. **Equal Rank Sum:**
   $$\text{Score}_{\text{Total}}(i, c) = \text{Rank}_{\text{SVM}}(i, c) + \text{Rank}_{\text{Qwen}}(i, c) + \text{Rank}_{\text{DeBERTa}}(i, c) + 10^{-6} \cdot \widetilde{\text{Margin}}_{\text{SVM}}(i, c)$$
   Where the continuous normalized SVM margin breaks any exact rank ties.

---

## 5. Ready-to-Run Submission Artifacts

Two submission files are generated and tested:
1. **Self-Contained Python Script:** [`notebooks/kaggle_tri_ensemble.py`](file:///d:/NameThatLLM/notebooks/kaggle_tri_ensemble.py) (can be pasted directly into a single Kaggle code cell).
2. **Jupyter Notebook:** [`notebooks/kaggle_tri_ensemble.ipynb`](file:///d:/NameThatLLM/notebooks/kaggle_tri_ensemble.ipynb) (can be uploaded via **File $\rightarrow$ Upload Notebook** on Kaggle).

### Submission Verification Checklist:
- [x] Evaluates 5-fold prompt-grouped cross-validation using DSU duplicate graph isolation.
- [x] Prints individual OOF Balanced Accuracy for CPU SVM, Qwen alone, DeBERTa alone, and the Tri-Model Blend.
- [x] Refits on 100% of labeled training data.
- [x] Predicts dynamically on `test.csv` (handles both the 3 interactive dummy rows and the ~4,000 hidden test rows).
- [x] Validates format compliance (columns `['id', 'model']`, no nulls, valid 18 official model strings).
- [x] Saves binary Apache Parquet to `/kaggle/working/submission.parquet`.

