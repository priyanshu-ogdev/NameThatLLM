# Name That LLM: Stylometric Fingerprint Analysis & Error Patterns

## 1. The Science of LLM Stylometry

In large language models, the text generated in response to an open prompt carries distinct statistical and structural "fingerprints." Even when multiple frontier models answer identical prompts with factual accuracy, subtle stylistic signatures emerge from five distinct layers of the generative stack:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Tokenizer Vocabulary & Segmentation Boundaries          │
│    (Byte-pair encoding biases, token splits on punctuation) │
├─────────────────────────────────────────────────────────────┤
│ 2. Post-Training Alignment (RLHF / DPO / Constitutional AI) │
│    (Tone, conversational hedging, safety preambles)         │
├─────────────────────────────────────────────────────────────┤
│ 3. Punctuation & Typography Habits                          │
│    (Em-dash —, en-dash –, straight quotes ", curly quotes “)│
├─────────────────────────────────────────────────────────────┤
│ 4. Structural Formatting & Markdown Preferences             │
│    (Headers ###, bold text **, bullet lists *, line breaks)  │
├─────────────────────────────────────────────────────────────┤
│ 5. Syntactic Complexity & Function Word Density             │
│    (Use of 'indeed', 'furthermore', 'however', 'perhaps')   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Key Stylometric Features & Extraction Engineering

A comprehensive set of stylometric features captures these variations without relying on the specific topic vocabulary:

### A. Length & Structural Mechanics
- **Character Count & Word Count:** Log-transformed character count $\log(1 + N_c)$ and word count $\log(1 + N_w)$.
- **Sentence Count & Length Dispersion:** Mean sentence length, standard deviation, and median. Certain models (like Claude Opus) exhibit longer, subordinate-clause-heavy sentences, whereas Flash models produce concise, punchy sentences.
- **Word Length Statistics:** Average characters per word, proportion of short words ($\le 3$ characters), and proportion of long words ($\ge 8$ characters).

### B. Typography & Punctuation Distributions
- **Dash Usage:** Distinctive divergence between em-dashes (`—`), en-dashes (`–`), and standard hyphens (`-`). Anthropic and OpenAI frontier models employ em-dashes frequently for rhetorical pauses.
- **Quotation Mark Preferences:** Straight ASCII quotes (`"` and `'`) vs. typographical curly quotes (`“`, `”`, `‘`, `’`).
- **Parentheses, Brackets, & Colons:** Rates of `(`, `)`, `[`, `]`, `:`, and `;`.
- **Special Formatting Characters:** Markdown indicators like `#`, `*`, `\n`, `\t`, `%`, and `/`.
- **Non-ASCII & Uppercase Density:** Fraction of uppercase characters (capitalization habits for titles) and non-ASCII character frequencies.

### C. Function Word & Contraction Densities
Content words (nouns, topical verbs) carry topic information, whereas **function words** reflect invariant grammatical styling:
- Conjunctions and transitions: `however`, `therefore`, `thus`, `moreover`, `furthermore`, `indeed`, `perhaps`, `although`, `because`.
- Modal verbs: `might`, `could`, `would`, `should`, `may`, `must`, `can`, `will`.
- Pronouns: `we`, `you`, `i`, `it`, `they`, `this`, `that`.
- Contraction frequency: Regex extraction of `\b\w+['’]\w+\b` (e.g., `don't`, `it's`, `we've`). High contraction usage characterizes conversational Claude and GPT variants, while technical Gemini outputs frequently use uncontracted formal forms.

---

## 3. Empirical Class Recalls & Baseline Accuracy

Analysis of 5-fold prompt-grouped cross-validation using the optimized Multiview Linear SVM baseline reveals stark differences in recognizability across the 18 models:

| Model Label | CV Recall | Recognizability Tier | Dominant Fingerprint Characteristics |
| :--- | :---: | :---: | :--- |
| **`gemini-3.1-pro-preview`** | **80.67%** | Tier 1 (Very High) | Strongest signature; structured bullet points, clear headings, formal tone. |
| **`claude-opus-5`** | **66.67%** | Tier 1 (Very High) | Highly reflective voice, nuanced punctuation, distinctive conversational tone. |
| **`gpt-6-astra`** | **66.00%** | Tier 1 (Very High) | Sophisticated syntax, distinctive satire and conceptual framing. |
| **`claude-opus-4-6`** | **55.33%** | Tier 2 (High) | Complex sentence constructions, balanced pacing. |
| **`qwen3.8-max`** | **48.00%** | Tier 2 (High) | Distinct technical lexicon, concise explanations. |
| **`gemini-3.7-flash`** | **43.33%** | Tier 3 (Moderate) | Clean formatting; high confusion with 3.8-flash. |
| **`gemini-3.6-flash`** | **42.67%** | Tier 3 (Moderate) | Direct answers, concise sentences. |
| **`gemini-3.8-flash`** | **39.33%** | Tier 3 (Moderate) | Multilingual prompts, structured bullet lists. |
| **`claude-sonnet-5`** | **38.67%** | Tier 3 (Moderate) | Pragmatic, highly organized, balanced tone. |
| **`gpt-5.5`** | **38.00%** | Tier 3 (Moderate) | Standard OpenAI baseline style; confused with Qwen and Kimi. |
| **`gpt-5.6-luna`** | **30.67%** | Tier 4 (Difficult) | Variant tuning; benefits strongly from neural representations. |
| **`kimi-k3`** | **25.33%** | Tier 4 (Difficult) | Chinese-English bilingual phrasing; improved by Qwen backbone. |
| **`claude-opus-4-8`** | **21.33%** | Tier 4 (Difficult) | Sits between Opus 4.6 and Opus 5; subtle boundary shifts. |
| **`gpt-5.6-sol`** | **18.00%** | Tier 5 (Very Hard) | Heavy intra-family confusion with Terra and Luna. |
| **`glm-5.3`** | **18.00%** | Tier 5 (Very Hard) | Confused with GLM-5.3-flash and Claude-Opus-5. |
| **`gpt-5.6-terra`** | **17.33%** | Tier 5 (Very Hard) | Extremely close to Luna and Sol; minimal distinguishing vocabulary. |
| **`glm-5.3-flash`** | **14.00%** | Tier 5 (Very Hard) | High intra-family overlap with base GLM-5.3. |
| **`claude-fable-5`** | **12.75%** | Tier 5 (Very Hard) | Lowest single-model recall; narrative prose mimics other Claude variants. |

---

## 4. Confusion Matrix Analysis: The Primary Error Modes

From the 5-fold confusion matrix, errors concentrate heavily into two categories:

### A. Severe Intra-Family Confusion (Within the Same Model Family)
1. **Google Gemini Flash Twins:**
   - `gemini-3.7-flash` misclassified as `gemini-3.8-flash`: **45 occurrences**
   - `gemini-3.8-flash` misclassified as `gemini-3.7-flash`: **42 occurrences**
   - *Insight:* Both models share near-identical tokenizers, system instruction habits, and length controls. Distinguishing them requires fine-grained character n-grams and subtle punctuation features.
2. **OpenAI GPT-5.6 Triplets:**
   - `gpt-5.6-terra` misclassified as `gpt-5.6-luna`: **31 occurrences**
   - `gpt-5.6-sol` misclassified as `gpt-5.6-terra`: **28 occurrences**
   - `gpt-5.6-sol` misclassified as `gpt-5.6-luna`: **27 occurrences**
   - *Insight:* Variants of GPT-5.6 differ primarily in fine-tuning weights rather than fundamental architectural changes.
3. **Zhipu GLM-5.3 Pair:**
   - `glm-5.3-flash` misclassified as `glm-5.3`: **27 occurrences**
   - `glm-5.3` misclassified as `glm-5.3-flash`: **25 occurrences**

### B. Cross-Family Confusion
1. **`glm-5.3` vs. `claude-opus-5` (30 occurrences):** Both models favor elaborate, well-structured, explanatory prose with similar sentence lengths.
2. **`gpt-5.5` vs. `qwen3.8-max` (27 occurrences):** Standard direct, helpful answers without strong stylistic embellishments.
3. **`kimi-k3` vs. `gpt-5.5` (25 occurrences):** Kimi's conversational style closely matches OpenAI's RLHF alignment.
4. **`claude-fable-5` vs. `claude-opus-5` (23 occurrences):** Narrative Claude output shares vocabulary and phrasing with Opus 5.

---

## 5. Strategic Takeaways for Modeling

1. **Hierarchical / Two-Stage Classification:** Because family identification (OpenAI vs. Google vs. Anthropic vs. GLM/Qwen) is substantially easier than intra-family separation, two-stage modeling or multi-task auxiliary loss (predicting lab + predicting model) can clarify boundary confusions.
2. **Complementarity of Ensembles:** Classical TF-IDF excels at exact character sequences and punctuation markers (scoring 80.7% on Gemini 3.1 Pro), whereas pretrained neural representations (like Qwen-1.7B) capture semantic context and subtle tone (boosting `kimi-k3` recall by +11.3% and `gpt-5.6-luna` by +10.0%).
