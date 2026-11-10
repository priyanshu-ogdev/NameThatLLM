# 18 models, one wide format

External responses for [Name That LLM](https://www.kaggle.com/competitions/name-that-llm), arranged as **`id` · `prompt` · 18 model-name columns**. One cell contains one answer; missing answers are null.

[**Dataset tour**](https://www.kaggle.com/code/olegzholobov/name-that-llm-wide-data-overview) · [**Seaborn NLP explorer**](https://www.kaggle.com/code/olegzholobov/name-that-llm-seaborn-nlp-explorer)

| View | Wide rows | Answer cells | Unique responses |
|---|---:|---:|---:|
| Full external corpus | 70,980 | 164,954 | **135,171** |
| 40–250-word candidates | 13,642 | 24,025 | **24,015** |

**1,902 exact user-prompt groups have responses from all 18 models.** Provenance retains 188,470 source occurrences. Multiple answers occupy extra rows; row positions are not matched generation runs.

## Start here

```python
from pathlib import Path
import pandas as pd
root = next(Path('/kaggle/input').rglob('external_short_wide.parquet')).parent
wide = pd.read_parquet(root / 'external_short_wide.parquet')
long = wide.melt(id_vars=['id', 'prompt'], var_name='model', value_name='text').dropna(subset=['text'])
```

Parquet tables, row/cell metadata, source URLs, model evidence, quality flags and checksums are included. See `README.txt` and `data_dictionary.json` for details.

## What to know

- **Original data stays separate.** Notebooks load the competition attachment and create its wide view in memory. Missing prompt wording remains null.
- **No full-response duplicates found** against 2,699 original training answers, including whitespace, Unicode/case and punctuation normalization. No complete normalized original answer occurs inside a longer external response. This does not rule out every paraphrase.
- **Quality varies.** Full data includes code, categorical, multimodal and agent outputs. Short candidates pass a mechanical filter; source evidence and missing finish metadata still matter.
- **Luna quarantine:** 74,790 cells with weaker, author-asserted labels, separate and excluded from default analysis.
- **Validate by prompt group**, keeping duplicate responses and related trajectories together. Shared user prompts can have different system/history settings.

Round 3 adds **14,830 unique responses and 497 short candidates**, including Artificial Analysis, Novigens, APIFlow, JetBrains and RISE releases. Earlier sources include Speechmap, public Hugging Face datasets and EQ-Bench.

**Other** reflects mixed source licenses; underlying content is not relicensed. Consult provenance and [competition rules](https://www.kaggle.com/competitions/name-that-llm/rules). Snapshot: **15 October 2026**.


## Files and joins

- `external_all_wide.parquet`: full external research view.
- `external_short_wide.parquet`: 40–250 whitespace-word candidates. The filter excludes flagged truncation, visible reasoning, cross-label shared text, explicitly unsuitable and no-training records; missing finish metadata remains flagged rather than certified complete.
- `external_cell_metadata.parquet`: join on `(view, id, model)` to recover `cell_id`, `response_id` and `prompt_group_id`.
- `external_cell_provenance.parquet`: every source occurrence. Join `cell_id` to `cell_id_all` or `cell_id_short`. Records retain source URL, requested/returned model identifiers, evidence, context hashes, quality flags and license declarations.
- `external_response_metadata.parquet`: one canonical metadata record per response ID.
- `external_row_metadata.parquet`: prompt-group identity, verified prompt status, response display rank and coverage.
- `quarantine_luna_*`: separate weak-label corpus and associated metadata/provenance.
- `source_catalog.csv`, `data_dictionary.json`, `validation.json`, `manifest.json`: source inventory, schema, checks and SHA-256 hashes.

Counts use different units: a response can be associated with multiple prompts and source occurrences. The all and short views are overlapping views of the same corpus and must not be concatenated as independent data. For text distributions, deduplicate by model and normalized response; for training, retain provenance to detect leakage across sources.

Only byte-exact, verified, nonempty user prompts are grouped across models. Unverified prompt occurrences stay isolated. Multiple different answers from one model are preserved in additional rows with deterministic display ranks. All and short views rank independently. Matching user prompts alone do not establish matching system prompts, history, images or decoding parameters. For strict comparisons, inspect occurrence context hashes and task IDs.

The release uses Parquet to preserve nulls and literal strings such as `NA`. Three Luna prompt strings contain NUL characters, preserved losslessly. Its noncanonical source label is author-asserted, and the last transcript item may be a tool result or intermediate message.

The original train file supplies prompt IDs, answers and model labels, but no prompt wording. Its local/notebook wide view has 150 prompt rows and 2,699 answers; group `p002` lacks a Fable answer. It is not redistributed here, following competition rule 2.4.b. Attach the competition to reproduce the original-data analysis after accepting its rules.

Overlap checks compare every available original train answer and the three supplied unlabeled test rows against the full external corpus. Exact bytes, whitespace, Unicode/case and punctuation-normalized comparisons found no matches, including across different model labels. Complete normalized original answers were also checked as substrings in longer external responses. Candidate-based character-similarity checks found no matches at 90% or 95%; retrieval requires three shared word trigrams and is not an exhaustive paraphrase search. Separate Luna data passed exact/normalized and substring checks; fuzzy checks do not cover that quarantine.

Model identity follows source-published labels and metadata, not independent authentication of every API response. Source licenses and unknown-license flags are retained; inclusion does not imply competition approval or override upstream terms. No test labels, predictions or submission are included.
