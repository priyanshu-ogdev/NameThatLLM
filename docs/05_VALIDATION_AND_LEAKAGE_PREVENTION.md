# Name That LLM: Validation Strategy & Leakage Prevention

## 1. The Prompt Topic Leakage Trap

In authorship attribution competitions where multiple models respond to shared prompts, the single most dangerous source of metric invalidation is **prompt-level data leakage**.

### Why Standard Random K-Fold Fails
If standard random K-fold cross-validation is used:
- Passages answering prompt $P_i$ appear in both the training partition and the validation partition.
- A model can easily achieve an artificially inflated cross-validation score (e.g. 50%–60% with naive bag-of-words) simply by memorizing which words appear in which prompt topic.
- However, on the **real hidden test set, prompt topics are completely unseen**. A model that relied on topic memorization completely collapses on the leaderboard.

---

## 2. Mandatory Protocol: StratifiedGroupKFold on `prompt_id`

To faithfully simulate evaluating on unseen prompts:
1. **Grouping by Prompt:** All 18 model responses answering the same `prompt_id` must reside **strictly in the same fold**.
2. **Stratification across Classes:** Every validation fold must contain representative samples of all 18 target model classes.
3. **Partition Parameters:** 5 folds, where each fold contains exactly 30 prompts (`150 / 5 = 30`), ensuring `30 × 18 = 540` passages per validation fold.

```
                           All 150 Prompts (2,699 Passages)
                                          │
            ┌──────────────┬──────────────┼──────────────┬──────────────┐
            ▼              ▼              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
       │ Fold 0   │   │ Fold 1   │   │ Fold 2   │   │ Fold 3   │   │ Fold 4   │
       │30 Prompts│   │30 Prompts│   │30 Prompts│   │30 Prompts│   │30 Prompts│
       │540 Pass. │   │540 Pass. │   │540 Pass. │   │540 Pass. │   │539 Pass.*│
       └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
       *Fold 4 contains 539 passages due to claude-fable-5 missing one prompt.
```

---

## 3. Duplicate Passage Isolation via Disjoint Set Union (DSU)

Even with prompt grouping, duplicate or near-duplicate passages across distinct prompt IDs can bridge folds. 

To eliminate this vulnerability:
1. Compute a normalized hash of each text passage:
   $$\text{key} = \text{SHA256}(\text{" ".join}(\text{text.split}()))$$
2. Build a graph where vertices are `prompt_id`s and edges connect prompts that share an identical passage hash.
3. Find the connected components using Disjoint Set Union (DSU).
4. Assign all prompts in the same connected component to the exact same validation fold.

---

## 4. Strict Pipeline Isolation: Zero Leakage Fitting

To maintain complete methodological integrity:
- **Never fit vectorizers on the full dataset before splitting:** Every `TfidfVectorizer`, token vocabulary, subword index, and `StandardScaler` must be fitted **exclusively on the training rows of the current fold**.
- **Out-of-Fold (OOF) Stacking Isolation:** If training a meta-model or stacker, the meta-features must be generated using an **inner grouped K-fold** inside the outer training fold, ensuring the meta-learner never sees outer validation labels during meta-training.
- **Fixed Reproducible Checksums:** All experimental runs must verify against fixed cryptographic hashes:
  - `train.csv` SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
  - `folds.csv` SHA-256: `d019f60a2cfaab61dcbb73f64a8c126b0871ff037c23142ef45b81c4cdab6804`

---

## 5. Verification Implementation Code

The canonical implementation is located in [`src/ntl/data.py`](file:///d:/NameThatLLM/src/ntl/data.py):

```python
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pandas as pd

def make_folds(df, n_splits=5, seed=42):
    group_ids = groups(df)  # Computes DSU duplicate-aware connected prompt groups
    folds = np.full(len(df), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(df.text, df.model, group_ids)):
        # Verify that all 18 labels exist in both partitions
        assert len(set(df.model.iloc[train_idx])) == 18
        assert len(set(df.model.iloc[valid_idx])) == 18
        folds[valid_idx] = fold
        
    return pd.DataFrame({"id": df.id, "fold": folds})
```
