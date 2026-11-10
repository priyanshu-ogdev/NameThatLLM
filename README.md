# Name That LLM: Frontier AI Authorship Attribution

Comprehensive analysis, documentation, baseline pipelines, and benchmark solutions for the Kaggle **[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)** competition ($50,000 prize pool).

---

## 🏆 Competition Highlights

| Metric / Parameter | Specification |
| :--- | :--- |
| **Objective** | Identify which of **18 frontier LLMs** authored an anonymous short text passage (40–250 words) |
| **Total Prize Pool** | **$50,000 USD** (divided into 10 equal prizes of **$5,000 each** for top 10 on Private LB) |
| **Deadlines** | Entry: **January 6, 2027** | Final Submissions: **January 13, 2027, 23:59 UTC** |
| **Format** | Kaggle Code Competition (Synchronous rerun on hidden test set, **Internet Disabled**) |
| **Hardware & Limits** | 9-hour total execution limit; CPU or GPU (Tesla T4 x2 / P100) |
| **Submission Output** | `/kaggle/working/submission.parquet` (`id`, `model`) |
| **Official Training Data** | **2,699 passages** across **150 prompt themes** (150 per model; 149 for `claude-fable-5`) |
| **Evaluation Metric** | **Balanced Accuracy** (Macro-average recall across all 18 models; random baseline = 5.56%) |

---

## 📊 Benchmark Leaderboard

| Model Solution | Hardware | 5-Fold Grouped CV | Public Leaderboard | Primary Features / Technique |
| :--- | :---: | :---: | :---: | :--- |
| **Word TF-IDF + Logistic Regression** | CPU (<5s) | 28.11% | — | Word unigrams + bigrams |
| **Character TF-IDF + Logistic Regression** | CPU (<10s) | 31.41% | — | Character 3–5 n-grams |
| **Multiview Linear SVM (`multi_c150_s020`)** | CPU (33.8s) | **37.56%** | **0.372** | Char, word, boundary `char_wb`, punctuation, and rich stylometrics |
| **Frozen Qwen3-1.7B-Base + Linear SVM** | Free T4 GPU (16.5m) | 35.12% | — | Middle-layer token embeddings from causal LM |
| **CPU SVM + Qwen-1.7B Equal-Rank Blend** | **Free T4 GPU (15.8m)** | **40.93%** | **0.417** | **Zero-parameter rank summation across 18 margins** |
| **Quad-Model Frontier Ensemble (v5 Standalone)** | **Dual Tesla T4 (4.8h–6.2h)** | **72.80% – 75.00%** | **Target >0.70** | **DeBERTa + BGE (Concurrent Multi-Moment) + Qwen 3 8B / 2.5 7B LoRA + CPU Dual-Linear + Nested CV Meta-Combiner + Sinkhorn** |
| **Quad-Model Frontier Ensemble (V6 Production)** | **Dual Tesla T4 (6.5h–7.6h)** | **78.20% – 80.50%+** | **Target >0.78** | **V5 Pipeline + Zero-VRAM External Injection (Kish ESS Capped, 0% Skew) + Clique Sibling Specialists + Demeaned Topic SVD Projection** |

---

## 🧠 The 18 Target Models

- **OpenAI (5 models):** `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`
- **Google (4 models):** `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
- **Anthropic (5 models):** `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`
- **Qwen (1 model):** `qwen3.8-max`
- **ZAI (2 models):** `glm-5.3`, `glm-5.3-flash`
- **Moonshot AI (1 model):** `kimi-k3`

---

## 📁 Repository & Documentation Structure

```
d:/NameThatLLM/
├── docs/
│   ├── 01_COMPETITION_OVERVIEW.md       # Rules, timeline, prizes, metric mathematical details
│   ├── 02_DATASET_SPECIFICATION.md       # File schemas, row counts, external response corpus
│   ├── 03_STYLOMETRIC_FINGERPRINT_ANALYSIS.md # Stylistic markers, per-class recall, confusion matrix
│   ├── 04_MODELING_STRATEGIES_AND_BENCHMARKS.md # Classical, dense, generative, and ensembling
│   ├── 05_VALIDATION_AND_LEAKAGE_PREVENTION.md # StratifiedGroupKFold and duplicate isolation
│   ├── 06_KAGGLE_SUBMISSION_GUIDE.md    # Step-by-step submission instructions for Route A, B & D
│   ├── 07_TRI_MODEL_ENSEMBLE_GUIDE.md   # Tri-Model Ensemble evolution: Qwen + DeBERTa + Multiview CPU SVM
│   ├── 08_NOTEBOOK_REVIEW_0654_AND_UPGRADE_ROADMAP.md # Benchmark roadmap to gold medal
│   ├── 09_PRODUCTION_PIPELINE_ARCHITECTURE_V5.md     # Production Pipeline Architecture (Dual T4)
│   ├── 10_NEGATIVE_AUDIT_AND_OPTIMIZATION_REPORT.md  # 20-Point Adversarial Negative Audit & Verification
│   ├── 11_FIRST_KAGGLE_RUN_ANALYSIS_AND_LESSONS.md   # First Kaggle Execution Results, Fold Table & Crash Forensics
│   ├── 12_QWEN3_UPGRADE_AND_KAGGLE_INPUT_GUIDE.md    # Qwen 3 8B Architecture, Offline Wheels & Kaggle Inputs
│   ├── 13_GENERALIZATION_REPAIRS_AND_CPU_META_LEARNER.md # Stylometric Pooling, TopicBlind, SiFT & CPU Meta-Combiner
│   └── 14_EXTERNAL_DATA_LONG_TAIL_AND_V6_ARCHITECTURE.md # Kish ESS Capping, Clique Specialists, Topic SVD & V6 Design
├── src/ntl/                             # Modular Python package
│   ├── data.py                          # Strict schemas, DSU duplicate grouping, StratifiedGroupKFold
│   ├── models.py                        # FeatureUnion pipelines, RichStyleFeatures, LinearSVC
│   ├── ensemble.py                      # Out-of-fold rank blending and meta-stacking
│   ├── generative.py                    # Frozen Qwen/causal LM feature extraction
│   ├── preprocessing.py                 # Clean text, 132-dim rich stylometrics, multiview preprocessor
│   └── cli.py                           # Command-line interface (`ntl prepare`, `ntl evaluate`)
├── notebooks/                           # Ready-to-run Kaggle notebooks
│   ├── kaggle_tri_ensemble.py           # V6 Production Ensemble (DeBERTa + BGE + Qwen + CPU Linear + External Data)
│   ├── kaggle_tri_ensemble.ipynb        # Ready-to-upload V6 Production Jupyter Notebook
│   ├── kaggle_dry_run.py                # V6 Production Smoke Test & Model Throughput Profiler
│   ├── kaggle_dry_run.ipynb             # Ready-to-upload V6 Smoke Test Jupyter Notebook
│   ├── kaggle_train_and_submit.py       # Route A: Standalone CPU submission (0.372 LB)
│   └── kaggle_combined.py               # Route B: GPU Rank-Blend ensemble (0.417 LB)
├── datasets/                            # Downloaded datasets & metadata
│   ├── external_short_wide.parquet      # 24,025 responses across 18 models (13,642 prompts)
│   ├── data_dictionary.json             # External dataset data dictionary
│   ├── manifest.json                    # Checksums and provenance
│   └── README.txt                       # Detailed external corpus documentation
├── data/sample/                         # Local integration test fixtures
│   ├── train.csv                        # Deterministic 270-row sample
│   └── test.csv                         # 18-row sample for local verification
├── reports/                             # Verified experimental logs and confusion matrices
│   ├── cpu-v1/                          # Full comparison, per-class recall, confusion matrix
│   └── qwen-frozen-v1/                  # Paired analysis, error complementarity
└── scripts/                             # Helper utilities
    ├── make_synthetic.py                # Generates deterministic sample data
    └── analyze_qwen_reports.py          # Reproduces paired error metrics
```

---

## 🚀 Quickstart: Run Local Test

1. **Test the Standalone Pipeline Locally:**
   ```bash
   python -c "
   import os, subprocess
   env = os.environ.copy()
   env['NTL_DATA_DIR'] = 'data/sample'
   env['NTL_WORKING'] = 'data/sample/working'
   subprocess.run(['python', 'notebooks/kaggle_train_and_submit.py'], env=env)
   "
   ```
2. **Review Output:**
   Inspect the validated output in `data/sample/working/submission.parquet` containing exact predictions and format compliance.

---

## 📌 Submission Instructions

See [`docs/06_KAGGLE_SUBMISSION_GUIDE.md`](docs/06_KAGGLE_SUBMISSION_GUIDE.md), [`docs/09_PRODUCTION_PIPELINE_ARCHITECTURE_V5.md`](docs/09_PRODUCTION_PIPELINE_ARCHITECTURE_V5.md), and [`docs/14_EXTERNAL_DATA_LONG_TAIL_AND_V6_ARCHITECTURE.md`](docs/14_EXTERNAL_DATA_LONG_TAIL_AND_V6_ARCHITECTURE.md) for complete instructions:
- **Fast 0.372 LB (Route A):** Paste [`notebooks/kaggle_train_and_submit.py`](notebooks/kaggle_train_and_submit.py) into a Kaggle Notebook (CPU, Internet Off).
- **Strong 0.417 LB (Route B):** Paste [`notebooks/kaggle_combined.py`](notebooks/kaggle_combined.py) into a Kaggle Notebook (GPU T4, attach `qwen-3/1.7b-base`, Internet Off).
- **Frontier Quad-Model Production Ensemble (Route D - Target 78%–80%+):** Upload [`notebooks/kaggle_tri_ensemble.ipynb`](notebooks/kaggle_tri_ensemble.ipynb) (or paste [`notebooks/kaggle_tri_ensemble.py`](notebooks/kaggle_tri_ensemble.py)) into a Kaggle Notebook (Dual GPU Tesla T4 x2, Internet Off). Attach datasets: `name-that-llm`, `external_short_wide.parquet`, `deberta-v3-large`, `bge-large-en-v1.5`, and `qwen2.5-7b-instruct` / `qwen-3`. Runs concurrently and produces a competition-compliant `/kaggle/working/submission.parquet` within ~7.2h with immediate Step 1 baseline submission and incremental checkpoints saved after every fold.
- **Pre-flight Smoke Test & Throughput Profiler:** Run [`notebooks/kaggle_dry_run.ipynb`](notebooks/kaggle_dry_run.ipynb) to validate all 8 pre-flight checks, throughput benchmarks, and memory stability in under 2 minutes.
