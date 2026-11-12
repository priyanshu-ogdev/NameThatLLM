# Name That LLM: Kaggle Submission & Reproducibility Guide

## 1. Quick Start Submission Overview

Submissions to Kaggle must run as self-contained notebooks with **Internet Off**, reading `/kaggle/input/competitions/name-that-llm/test.csv` and writing `/kaggle/working/submission.parquet`.

Two validated submission approaches are provided:

```
┌───────────────────────────────────────────────────────────────┐
│ Route A: Standalone CPU Model (Fast, Zero Dependencies)       │
│ • Notebook: notebooks/kaggle_train_and_submit.py              │
│ • Accelerator: None (CPU)  | Runtime: ~30 seconds             │
│ • External Datasets: None  | Public LB: 0.372                 │
├───────────────────────────────────────────────────────────────┤
│ Route B: GPU Multimodal Rank-Blend (Fast Baseline)            │
│ • Notebook: notebooks/kaggle_combined.py                      │
│ • Accelerator: Free GPU T4 | Runtime: ~15.8 minutes           │
│ • External Model: Qwen3-1.7B-Base | Public LB: 0.417         │
├───────────────────────────────────────────────────────────────┤
│ Route D: Frontier Quad-Model Production Pipeline (V6 Gold)    │
│ • Notebook: notebooks/kaggle_tri_ensemble.ipynb / .py         │
│ • Accelerator: Dual Tesla T4 (GPU T4 x2) | Runtime: ~4.9–5.8h  │
│ • Inputs: DeBERTa-v3 + BGE-large + Qwen 3 8B / 2.5 + External │
│ • Target Score: 0.780 – 0.805+ Macro Balanced Accuracy        │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Route A: Standalone CPU Submission (0.372 LB)

This approach trains the Multiview Linear SVM directly inside the Kaggle notebook environment using preinstalled packages (`scikit-learn`, `pandas`, `pyarrow`). It requires no extra datasets, no model weights, and zero GPU quota.

### Step-by-Step Instructions:
1. Open the pasteable Python file: [`notebooks/kaggle_train_and_submit.py`](file:///d:/NameThatLLM/notebooks/kaggle_train_and_submit.py).
2. Copy the entire file contents.
3. On Kaggle, navigate to the competition page: [Name That LLM Code](https://www.kaggle.com/competitions/name-that-llm/code).
4. Click **New Notebook**.
5. Paste the copied code into a single code cell.
6. Verify notebook settings:
   - **Accelerator:** `None` (CPU)
   - **Internet:** `Off`
   - **Inputs:** Ensure `Name That LLM` competition data is attached under `/kaggle/input/competitions/name-that-llm/`.
7. Click **Run All** once to verify interactive execution. The output will print:
   ```
   Training passages: 2699; prompts: 150; test passages: 3
   Full-data model trained in 33.8 seconds. No GPU was used.
   SUCCESS: validated submission saved to /kaggle/working/submission.parquet
   Submission rows: 3
   ```
8. Click **Save Version** -> Choose **Save & Run All (Commit)**.
9. Once the commit finishes, click on the completed notebook version, find `submission.parquet`, and click **Submit to Competition**.
10. Check **My Submissions** to confirm the **0.372 public balanced accuracy score**.

---

## 3. Route B: GPU Multimodal Rank-Blend (0.417 LB)

This approach extracts middle-layer representations from a frozen generative model (`Qwen3-1.7B-Base`) and blends them with the CPU Multiview SVM using within-passage rank summation.

### Step-by-Step Instructions:
1. Open the competition [Code tab](https://www.kaggle.com/competitions/name-that-llm/code) and click **New Notebook**.
2. Open [`notebooks/kaggle_combined.py`](file:///d:/NameThatLLM/notebooks/kaggle_combined.py) and copy its entire contents into the notebook cell (or import [`notebooks/kaggle_combined.ipynb`](file:///d:/NameThatLLM/notebooks/kaggle_combined.ipynb)).
3. Under the notebook's **Add Input** sidebar:
   - Add competition dataset: `Name That LLM`.
   - Add Kaggle public model: search for `qwen-lm/qwen-3` -> select **Transformers** -> **1.7b-base** -> **version 1** ([Model link](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/1.7b-base/1)).
4. Configure notebook settings:
   - **Accelerator:** `GPU T4 x2` (or `GPU P100`)
   - **Internet:** `Off`
5. Click **Run All**. The notebook:
   - Audits the official training passages and verifies fold checksums.
   - Extracts middle-layer mean pooled token features with Qwen3-1.7B.
   - Fits the linear SVM and evaluates out-of-fold accuracy.
   - Fits the CPU Multiview SVM.
   - Computes the equal-weight rank blend.
   - Generates and validates `/kaggle/working/submission.parquet`.
   - Execution finishes in approximately **15.85 minutes**.
6. Click **Save Version** -> **Save & Run All (Commit)**.
7. Upon successful execution, submit the output `submission.parquet`.
8. The submission achieves **0.417 public balanced accuracy**.

---

## 4. Route D: Frontier Quad-Model Production Pipeline (V6 Gold Medal Target >0.78)

This is the flagship production ensemble combining **DeBERTa-v3-large** (disentangled attention, GPU 0), **BGE-large-en-v1.5** (contrastive bi-encoder, GPU 1), **Qwen 3 8B / 2.5 7B LoRA** (generative causal LLM with Spike & Gap Hierarchical Head, sharded across dual GPUs), **CPU Multiview Dual-Linear** (~850k n-grams, 145 stylometrics expanded to 1,835 Poly-2/RBF dimensions, and Intra-Family Clique Specialists), and **24,025 Zero-VRAM external passages** (`external_short_wide.parquet`).

### Step-by-Step Instructions:

#### 1. Pre-flight Smoke Test (Recommended First Step)
Before committing the full 5.8-hour run, execute the pre-flight verifier to benchmark throughput and confirm hardware stability:
1. Upload [`notebooks/kaggle_dry_run.ipynb`](file:///d:/NameThatLLM/notebooks/kaggle_dry_run.ipynb) (or paste [`notebooks/kaggle_dry_run.py`](file:///d:/NameThatLLM/notebooks/kaggle_dry_run.py)) to a new Kaggle notebook.
2. Select **Accelerator: GPU T4 x2**, **Internet: Off**.
3. Attach the datasets listed below.
4. Click **Run All**. The dry run executes 9 comprehensive pre-flight checks in **< 2 minutes**, printing the Wall-Clock Feasibility Matrix and Go/No-Go verdict.

#### 2. Full Production Run Setup
1. On Kaggle, open the competition [Code tab](https://www.kaggle.com/competitions/name-that-llm/code) and select **New Notebook** (or upload [`notebooks/kaggle_tri_ensemble.ipynb`](file:///d:/NameThatLLM/notebooks/kaggle_tri_ensemble.ipynb)).
2. If pasting manually, copy the entire contents of [`notebooks/kaggle_tri_ensemble.py`](file:///d:/NameThatLLM/notebooks/kaggle_tri_ensemble.py) into the code cell.
3. In the right-hand **Notebook Settings** panel:
   - **Accelerator:** `GPU T4 x2` (Dual Tesla T4).
   - **Internet:** `Off` (strict competition requirement).
4. Under the **Add Input** sidebar, attach the following **6 datasets**:

| Input Item | Kaggle Search Term / Dataset Slug | Mounted Directory | Role in V6 Pipeline |
| :--- | :--- | :--- | :--- |
| **1. Competition Data** | `Name That LLM` | `/kaggle/input/competitions/name-that-llm` | Official `train.csv` (2,699 rows) and hidden `test.csv` |
| **2. External Corpus** | `name-that-llm-wide-data-overview` | `/kaggle/input/.../external_short_wide.parquet` | 24,025 passages with Kish ESS variance capping |
| **3. Qwen 3 8B / 2.5 7B** | `qwen-3` (or `qwen2.5-7b-instruct`) | `/kaggle/input/.../qwen-3/...` | Generative causal backbone with Spike & Gap Head |
| **4. DeBERTa-v3-large** | `deberta-v3-large` | `/kaggle/input/.../deberta-v3-large` | 435M Disentangled attention encoder (GPU 0) |
| **5. BGE-large-en-v1.5** | `bge-large-en-v1.5` | `/kaggle/input/.../bge-large-en-v1.5` | 335M RetroMAE contrastive bi-encoder (GPU 1) |
| **6. Offline Wheels** | `transformers wheels` | `/kaggle/input/.../transformers*.whl` | Offline dependencies for `transformers >= 4.49.0` |

#### 3. Execution Pipeline Stages:
When executed, the pipeline sequentially advances through 5 clean stages:
- **STEP 0 (Multi-Model Upfront Pre-Tokenization & RAM Caching):** Precomputes token IDs, attention masks, and special `[NL]` structure across all models once in parallel (~28 MB host RAM, saving 100k+ tokenizations).
- **STEP 1 (CPU Multiview & Immediate Baseline Submission):** Fits the FeatureUnion (~850k n-grams + 145 stylometrics) and writes a valid, non-empty `submission.parquet` within the first **30 seconds**.
- **STEP 2 (5-Fold Prompt-Grouped Cross-Validation):**
  - **Parallel GPU Execution:** DeBERTa (GPU 0) and BGE (GPU 1) train simultaneously via `ThreadPoolExecutor(max_workers=2)`.
  - **Qwen LoRA (Dual GPU):** Sharded across both GPUs (`device_map="auto"`), extracting multi-layer hidden states (-1, -5, -9) through the `SpikeAndGapHierarchicalHead`.
  - **CPU Dual-Linear with External Injection:** Trains LinearSVC + Ridge on the combined competition + external corpus with Kish ESS variance capping ($\le 0.35$), regularizer invariance, moment alignment, and Intra-Family Clique Specialist tie-breakers.
  - **Incremental Checkpoints:** Updates and saves a valid `submission.parquet` after **every completed fold**.
- **STEP 3 (Out-of-Fold Evaluation, Calibration & Meta-Combiner):** Evaluates stream OOF scores, fits NLL-optimal Platt temperatures, and fits the PyTorch CPU Autograd Meta-Combiner using nested 5-fold cross-validation.
- **STEP 4 (Bagged Test Predictions, Sinkhorn & Final Submission):** Computes bagged test predictions across all folds, applies Sinkhorn-Knopp probability matrix balancing to enforce uniform test marginals, and atomically replaces `/kaggle/working/submission.parquet`.

#### 4. Save and Submit:
1. Click **Save Version** -> **Save & Run All (Commit)**.
2. The notebook runs in approximately **4.92–5.78 hours** (295.2 min Dual-GPU concurrent / 346.8 min conservative upper bound with 2x TTA), leaving **+2.42h to +3.28h** below the 8.2-hour internal guard and **+3.22h to +4.08h** below Kaggle's 9.0-hour wall-clock timeout.
3. Once completed, navigate to the notebook version, locate `submission.parquet`, and click **Submit to Competition**.

---

## 5. Submission Validation Checklist

Before submitting, verify that the notebook satisfies all platform requirements:

- [x] **File Path:** Saved to `/kaggle/working/submission.parquet`.
- [x] **File Format:** Binary Apache Parquet (`pd.DataFrame.to_parquet('submission.parquet', index=False)`).
- [x] **Column Names:** Exactly `id` and `model`.
- [x] **Row Count:** Matches `len(test_df)` dynamically (handles 3 rows during testing, ~4,000 during scoring).
- [x] **ID Integrity:** Preserves original string types and leading zeros; IDs match `test['id'].tolist()` in order.
- [x] **Label Validity:** Every predicted string is an exact match for one of the 18 official model names:
  `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`, `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`, `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`, `qwen3.8-max`, `glm-5.3`, `glm-5.3-flash`, `kimi-k3`.
- [x] **Null Free:** Zero NaN, null, or empty string values in any prediction.
