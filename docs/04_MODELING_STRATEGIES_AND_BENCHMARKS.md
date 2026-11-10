# Name That LLM: Modeling Strategies, Benchmarks & Proven Solutions

## 1. Experimental Benchmark Landscape

Through systematic prompt-grouped cross-validation on the official 2,699 training passages, multiple modeling paradigms were tested and evaluated. Below is the comparative benchmark table:

| Model Architecture / Paradigm | Hardware Required | Grouped 5-Fold CV Balanced Accuracy | Public Leaderboard Balanced Accuracy | Notes & Key Takeaways |
| :--- | :---: | :---: | :---: | :--- |
| **Word TF-IDF + Logistic Regression** | CPU (< 5s) | 28.11% | — | Baseline bag-of-words; overfits to prompt topic keywords. |
| **Character TF-IDF + Logistic Regression** | CPU (< 10s) | 31.41% | — | Char 3-5 n-grams capture subwords and punctuation. |
| **Word + Char Logistic Regression** | CPU (< 15s) | 31.04% | — | Simple concatenation of word and character vectors. |
| **Word + Char Linear SVM** | CPU (< 20s) | 30.63% | — | Standard LinearSVC with balanced class weights. |
| **Word + Char + Basic Style SVM** | CPU (25s) | 36.67% | — | Adding length, digit, and basic punctuation rates yields +6.04 points. |
| **Boundary-Character / Punctuation SVM** | CPU (30s) | 36.12% | — | Focuses on `char_wb` (word-boundary character n-grams). |
| **Four Text Views + Style Linear SVM** | CPU (35s) | 37.12% | — | Combines char, word, boundary, and punctuation views. |
| **Tuned Multiview SVM (`multi_c150_s020`)** | **CPU (33.8s)** | **37.56%** | **0.372** | **Best standalone CPU model.** Zero external weights, fast execution. |
| **Frozen BGE-Small Retrieval Embeddings** | CPU / GPU (339s) | 11.67% – 16.22% | — | Retrieval models map semantics and erase stylometry. |
| **Frozen Qwen3-1.7B-Base (Middle Layer)** | Free GPU (16.5m) | 35.12% | — | Standalone neural feature extractor; captures latent syntax. |
| **CPU SVM + Frozen Qwen-1.7B Rank Blend** | **Free GPU (15.8m)** | **40.93%** | **0.417** | **Best submitted solution.** +4.5% public LB gain over CPU baseline. |

---

## 2. Deep Dive: The Winning Multiview CPU Architecture

The top-performing CPU model uses a high-dimensional **FeatureUnion** across five complementary stylometric representations, followed by a regularized `LinearSVC`:

```
                       ┌────────────────────────────────────────────────────────┐
                       │ Raw Text Passage (Case & Punctuation Intact)           │
                       └───────────────────┬────────────────────────────────────┘
                                           │
         ┌───────────────────┬─────────────┴───────┬──────────────────┬─────────────────┐
         ▼                   ▼                     ▼                  ▼                 ▼
   ┌───────────┐       ┌───────────┐         ┌───────────┐      ┌───────────┐     ┌───────────┐
   │ Char TFIDF│       │ Word TFIDF│         │Boundary WB│      │Punct TFIDF│     │Rich Style │
   │ (1-5 char)│       │ (1-2 word)│         │(3-5 char) │      │(tokens &  │     │(counts,   │
   │ 180k feat │       │ 150k feat │         │100k feat  │      │ symbols)  │     │ ratios)   │
   └─────┬─────┘       └─────┬─────┘         └─────┬─────┘      └─────┬─────┘     └─────┬─────┘
         │                   │                     │                  │                 │ (weight 0.2)
         └───────────────────┼─────────────────────┴──────────────────┴─────────────────┘
                             ▼
               ┌───────────────────────────┐
               │ FeatureUnion Concat Matrix│
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │ LinearSVC (C=1.5, balanced│
               │ dual=True, max_iter=15000)│
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │ 18 Continuous Margins     │
               └───────────────────────────┘
```

### Components of the FeatureUnion:
1. **Full Character N-Grams:** `TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2, max_features=180_000, sublinear_tf=True)`.
2. **Whitespace Word N-Grams:** `TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, max_features=150_000, sublinear_tf=True)`. Preserves attached punctuation marks rather than stripping them.
3. **Word-Boundary Characters:** `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100_000, sublinear_tf=True)`. Captures prefixes, suffixes, and boundary transitions.
4. **Punctuation & Token Vectorizer:** `TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)`. Explicitly isolates isolated punctuation marks as tokens.
5. **Rich Stylometrics with Scaler:** 50+ hand-engineered linguistic measurements scaled via `StandardScaler()`, downweighted by a multiplier of `0.2` to prevent dense features from dominating sparse n-grams.

---

## 3. Why Standard Sentence Transformers Fail

When testing pretrained retrieval encoders like `bge-small-en-v1.5`, the models achieved only **11.67% to 16.22% balanced accuracy** (barely above the 5.56% random baseline).

### The Stylometry vs. Semantic Embedding Paradox
- **Retrieval Embeddings Collapse Style:** Dense sentence encoders are fine-tuned on contrastive objectives to make paraphrases close in vector space. As a result, they aggressively discard surface punctuation, sentence length variance, and token cadences to map the *meaning* (the prompt topic).
- **Topic Overfitting:** If an encoder clusters passages by prompt theme rather than author model, the classifier fails to distinguish between the 18 models answering the exact same prompt.

---

## 4. Frozen Generative LLM Encoders (Qwen3-1.7B-Base)

In contrast to retrieval embeddings, **generative causal language models** preserve detailed sequential syntax:
- Model used: `Qwen/Qwen3-1.7B-Base` (Apache 2.0 license).
- Hidden layer representations: Rather than extracting from the final layer (which collapses into next-token prediction logits), extracting from **middle layers** (e.g. layer 14 out of 28) and taking the mean-pooled token vectors retains both syntactic abstraction and stylometric nuances.
- Feature extraction speed: 577 seconds for 2,699 passages on a single Kaggle Tesla T4 GPU.
- Classification on extracted vectors: Scaled with `StandardScaler()` followed by `SVC(kernel="linear", class_weight="balanced")`.
- Performance: **35.12% balanced accuracy standalone**.

---

## 5. The Breakthrough: Fixed Equal-Rank Ensembling

While the CPU SVM scored 37.56% and frozen Qwen scored 35.12%, error analysis revealed extraordinary **complementarity**:
- **Both models correct:** 531 passages
- **CPU model alone correct:** 483 passages
- **Qwen model alone correct:** 417 passages
- **Neither model correct:** 1,268 passages

Qwen correctly classified **24.75% of the errors made by the CPU model**.

```
                        ┌───────────────────────────────┐
                        │ Anonymous Test Passage        │
                        └──────────────┬────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
        ┌───────────────┐                             ┌───────────────┐
        │ CPU Multiview │                             │ Frozen Qwen   │
        │ TF-IDF + SVM  │                             │ 1.7B + SVM    │
        └───────┬───────┘                             └───────┬───────┘
                ▼                                             ▼
        ┌───────────────┐                             ┌───────────────┐
        │ 18 Continuous │                             │ 18 Continuous │
        │ Decision      │                             │ Decision      │
        │ Margins       │                             │ Margins       │
        └───────┬───────┘                             └───────┬───────┘
                ▼                                             ▼
        ┌───────────────┐                             ┌───────────────┐
        │ Within-Passage│                             │ Within-Passage│
        │ Rank (1..18)  │                             │ Rank (1..18)  │
        └───────┬───────┘                             └───────┬───────┘
                └──────────────────────┬──────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │ Equal-Weight Rank Summation   │
                       │ Rank_Total = Rank_CPU +       │
                       │              Rank_Qwen        │
                       │ (Ties broken by CPU margin)   │
                       └───────────────┬───────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │ Argmax -> Predicted Model     │
                       └───────────────────────────────┘
```

### Why Rank Summation Beats Probability Averaging
- Linear SVM margins are uncalibrated decision distances, not calibrated probabilities.
- Softmax scaling of SVM margins introduces arbitrary temperature sensitivity.
- **Rank transformation** normalizes the predictions within each passage to an invariant scale $[1, 18]$, eliminating scale disparity between the sparse TF-IDF model and dense neural model without fitting extra parameters on validation data.

### Resulting Performance
- **Grouped CV:** **40.93%** (a +3.37 percentage-point jump over CPU alone).
- **Public Leaderboard:** **0.417** (up from 0.372, confirming a +4.5 percentage-point boost on unseen Kaggle test data).

---

## 6. Next-Generation Experimentation Roadmap

1. **Scale Backbone to Qwen3-8B-Base / Qwen3.5-4B:**
   - Qwen-8B base parameters fit within a single 16 GB T4 GPU using FP16 / BF16.
   - Larger parameter models have richer representations of subtle author style.
2. **DeBERTa-v3-large Supervised Classifier:**
   - Bidirectional attention with disentangled position embeddings excels at token-level nuance.
   - Fine-tune with LoRA (Low-Rank Adaptation, rank=16, alpha=32) directly on prompt-grouped folds.
3. **Data Augmentation via Open External Response Corpus:**
   - Incorporate the 24,025 curated passages from `datasets/external_short_wide.parquet` to pre-train or fine-tune neural encoders on all 18 models.
