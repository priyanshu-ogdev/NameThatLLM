# Name That LLM: Dataset Specification & Schemas

## 1. Official Dataset Structure

The official competition dataset comprises the following primary files:

```
/kaggle/input/competitions/name-that-llm/
├── train.csv                   # Training set (2,699 passages with labels and prompt IDs)
├── test.csv                    # Evaluation set (3 visible rows interactive; ~4,000 hidden test rows)
└── sample_submission.parquet   # Reference submission format with schema validation
```

---

## 2. File Schemas & Column Definitions

### A. `train.csv` (Training Passages)
Contains 2,699 labeled passages generated across 150 distinct prompt themes.

| Column Name | Data Type | Nullable | Description & Constraints |
| :--- | :--- | :--- | :--- |
| **`id`** | `string` | No | Unique passage identifier (e.g., `"train-0001"`). Must preserve string type to avoid stripping leading zeros. |
| **`prompt_id`** | `string` | No | Identifier for the user prompt/topic that prompted this generation (150 distinct prompts). |
| **`text`** | `string` | No | The generated LLM passage. Word count ranges from 36 to 274 words (median: 101 words). |
| **`model`** | `string` | No | Ground-truth author label. Must be exactly one of the 18 official model names. |

### B. `test.csv` (Test Passages)
Contains unlabelled passages to be classified.

| Column Name | Data Type | Nullable | Description & Constraints |
| :--- | :--- | :--- | :--- |
| **`id`** | `string` | No | Unique passage identifier matching test instances. |
| **`text`** | `string` | No | The anonymous generated text passage to identify. |

### C. `sample_submission.parquet` / `submission.parquet`
The output submission must be written in Apache Parquet format.

| Column Name | Data Type | Nullable | Description & Constraints |
| :--- | :--- | :--- | :--- |
| **`id`** | `string` | No | Exactly matches every `id` present in `test.csv`. No duplicate or omitted IDs permitted. |
| **`model`** | `string` | No | Predicted author model name (exact string from the 18 official labels). |

---

## 3. The 18 Target Frontier Models

The 18 target classes represent state-of-the-art models from six leading AI research organizations:

| Provider / Lab | Model Label | Architecture Family | Notes & Distinguishing Characteristics |
| :--- | :--- | :--- | :--- |
| **OpenAI** (5 models) | `gpt-5.5` | Flagship GPT-5 series | Balanced conversational tone, standard formatting |
| | `gpt-5.6-luna` | GPT-5.6 variant | Variant fine-tuning, distinctive punctuation cadence |
| | `gpt-5.6-sol` | GPT-5.6 variant | Distinct lexical density, often confused with terra |
| | `gpt-5.6-terra` | GPT-5.6 variant | Variant fine-tuning; subtle intra-family markers |
| | `gpt-6-astra` | Next-gen Astra architecture | High recall (66%), distinctive reasoning and satire style |
| **Google** (4 models) | `gemini-3.1-pro-preview` | Gemini 3 Pro preview | Highest baseline recall (80.7%), bold headers, structured bullets |
| | `gemini-3.6-flash` | Gemini 3 Flash generation | Lightweight reasoning, distinct punctuation |
| | `gemini-3.7-flash` | Gemini 3.7 Flash generation | Heavily confused with gemini-3.8-flash |
| | `gemini-3.8-flash` | Gemini 3.8 Flash generation | Multilingual prompts, structured bullet lists |
| **Anthropic** (5 models) | `claude-fable-5` | Creative/narrative Claude | Lowest baseline recall (12.8%), 149 train rows |
| | `claude-opus-4-6` | Claude 4.6 Opus series | Strong stylistics (55.3% recall), nuanced clause structure |
| | `claude-opus-4-8` | Claude 4.8 Opus series | Nuanced voice, transitional phrasing |
| | `claude-opus-5` | Claude 5 Opus series | High baseline recall (66.7%), reflective and conversational tone |
| | `claude-sonnet-5` | Claude 5 Sonnet series | Concise, clean formatting, balanced voice |
| **Qwen / Alibaba** (1 model) | `qwen3.8-max` | Qwen frontier dense/MoE | Distinct technical vocabulary, high recall (48%) |
| **ZAI / Zhipu** (2 models) | `glm-5.3` | GLM-5 series | Often confused with GLM-5.3-flash and Claude-Opus-5 |
| | `glm-5.3-flash` | Lightweight GLM-5 variant | Lower recall (14%), concise phrasing |
| **Moonshot AI** (1 model) | `kimi-k3` | Kimi frontier model | Distinctive Chinese-English bilingual cadence, 25.3% recall |

---

## 4. Class Distribution & Training Passage Counts

The training set contains **2,699 passages**:
- Exactly **150 passages** for 17 models: `17 × 150 = 2,550` passages.
- Exactly **149 passages** for `claude-fable-5`: `149` passages.
- Total passages: `2,550 + 149 = 2,699` rows.
- Total prompt groups: **150 unique `prompt_id`s**.
- Prompt group `p002` is missing the `claude-fable-5` generation.

### Passage Length Summary
- **Minimum words:** 36
- **10th percentile:** 68 words
- **Median words:** 101 words
- **90th percentile:** 153 words
- **Maximum words:** 274 words

---

## 5. External Response Corpus (`olegzholobov/name-that-llm-response-corpus`)

To augment the relatively compact 2,699-row official training set, community members published an extensive external response corpus under open licensing, verified compatible with competition Rule 6:

- **Local Path in Workspace:** `d:/NameThatLLM/datasets/`
  - `external_short_wide.parquet` (13.5 MB): Filtered subset of passages meeting the 40–250 word length criteria. Contains **13,642 wide prompt rows** with **24,025 valid model response cells** across all 18 models.
  - `data_dictionary.json` & `manifest.json`: Full schema documentation and cryptographic SHA-256 hashes.
- **Deduplication Audit:** Exact, normalized, and fuzzy n-gram searches verify zero leakage or exact matches with the official 2,699 training passages.
- **Model Distribution in External Short Corpus:**
  - `claude-opus-4-6`: 4,911 responses
  - `gpt-5.6-sol`: 3,395 responses
  - `gpt-5.6-luna`: 2,285 responses
  - `kimi-k3`: 1,689 responses
  - `gpt-5.6-terra`: 1,544 responses
  - `claude-opus-4-8`: 1,428 responses
  - `claude-fable-5`: 1,148 responses
  - `gemini-3.1-pro-preview`: 1,141 responses
  - `claude-sonnet-5`: 1,133 responses
  - `gpt-5.5`: 1,111 responses
  - `glm-5.3-flash`: 1,056 responses
  - `claude-opus-5`: 828 responses
  - `qwen3.8-max`: 799 responses
  - `gpt-6-astra`: 723 responses
  - `glm-5.3`: 603 responses
  - `gemini-3.6-flash`: 83 responses
  - `gemini-3.7-flash`: 77 responses
  - `gemini-3.8-flash`: 71 responses
