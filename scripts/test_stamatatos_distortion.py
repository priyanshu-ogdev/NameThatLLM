import sys
sys.path.insert(0, ".")

import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from notebooks.kaggle_tri_ensemble import LABELS, make_prompt_groups, clean_text, DualLinearClassifier

# Function words list (top 150 English function words)
FUNCTION_WORDS = set((
    "a about above after again against all am an and any are aren't as at be because "
    "been before being below between both but by can can't cannot could couldn't did "
    "didn't do does doesn't doing don't down during each few for from further had "
    "hadn't has hasn't have haven't having he he'd he'll he's her here here's hers "
    "herself him himself his how how's i i'd i'll i'm i've if in into is isn't it "
    "it's its itself let's me more most mustn't my myself no nor not of off on once "
    "only or other ought our ours ourselves out over own same shan't she she'd she'll "
    "she's should shouldn't so some such than that that's the their theirs them "
    "themselves then there there's these they they'd they'll they're they've this "
    "those through to too under until up very was wasn't we we'd we'll we're we've "
    "were weren't what what's when when's where where's which while who who's whom "
    "why why's with won't would wouldn't you you'd you'll you're you've your yours "
    "yourself yourselves therefore however thus moreover furthermore indeed perhaps"
).split())

WORD_RE = re.compile(r"\b\w+(?:['\u2019\u2018]\w+)?\b")

def distort_text(text: str) -> str:
    """Stamatatos-style topic masking: replace content words with <CW>, preserving function words and punctuation."""
    t = clean_text(text)
    
    def repl(match):
        w = match.group(0)
        if w.lower() in FUNCTION_WORDS:
            return w
        return "<CW>"
    
    return WORD_RE.sub(repl, t)

# Test on a passage
sample_text = "Certainly! The Roman Empire expanded rapidly across Europe, conquering Gaul and Britain with disciplined legions."
print("Original text:")
print(" ", sample_text)
print("\nDistorted (Topic-Masked) text:")
print(" ", distort_text(sample_text))

train_df = pd.read_csv("data/sample/train.csv")
groups = make_prompt_groups(train_df)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_texts = train_df["text"].tolist()
distorted_texts = [distort_text(t) for t in all_texts]
y_str = train_df["model"].to_numpy()

# Compare standard Word TF-IDF vs Distorted Word TF-IDF across 5 folds
vec_orig = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
vec_dist = TfidfVectorizer(ngram_range=(1, 3), max_features=50000, sublinear_tf=True)

X_orig = vec_orig.fit_transform(all_texts)
X_dist = vec_dist.fit_transform(distorted_texts)

scores_orig = []
scores_dist = []
scores_combined = []

for tr_idx, va_idx in sgkf.split(all_texts, y_str, groups):
    # Model 1: Original word n-grams
    clf1 = LinearSVC(C=1.2, class_weight="balanced", dual="auto", max_iter=600, tol=1e-3, random_state=42)
    clf1.fit(X_orig[tr_idx], y_str[tr_idx])
    pred1 = clf1.predict(X_orig[va_idx])
    scores_orig.append(balanced_accuracy_score(y_str[va_idx], pred1))

    # Model 2: Distorted (topic-masked) n-grams
    clf2 = LinearSVC(C=1.2, class_weight="balanced", dual="auto", max_iter=600, tol=1e-3, random_state=42)
    clf2.fit(X_dist[tr_idx], y_str[tr_idx])
    pred2 = clf2.predict(X_dist[va_idx])
    scores_dist.append(balanced_accuracy_score(y_str[va_idx], pred2))

    # Model 3: Soft margin blend of Orig + Distorted
    m1 = clf1.decision_function(X_orig[va_idx])
    m2 = clf2.decision_function(X_dist[va_idx])
    pred_blend = clf1.classes_[np.argmax(m1 + m2, axis=1)]
    scores_combined.append(balanced_accuracy_score(y_str[va_idx], pred_blend))

print(f"\n5-Fold CV on Sample Data:")
print(f"  Raw Word TF-IDF:               {np.mean(scores_orig)*100:.2f}%")
print(f"  Topic-Masked Word TF-IDF:      {np.mean(scores_dist)*100:.2f}%")
print(f"  Blended (Raw + Masked):        {np.mean(scores_combined)*100:.2f}%")

