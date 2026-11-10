import numpy as np
import scipy.stats
import re

# 1. Test rankdata vectorized
m = np.random.randn(10, 18)
r1 = np.apply_along_axis(scipy.stats.rankdata, 1, m)
r2 = scipy.stats.rankdata(m, axis=1)
assert np.allclose(r1, r2)
print("scipy.stats.rankdata(m, axis=1) works natively!")

# 2. Check CONTRACTION_RE in kaggle_tri_ensemble.py
with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    code = f.read()

import re
matches = re.findall(r"CONTRACTION_RE = .*", code)
print("CONTRACTION_RE:", matches)

# 3. Check RichStyleFeatures output dim
from notebooks.kaggle_tri_ensemble import RichStyleFeatures
feat_extractor = RichStyleFeatures()
sample_texts = ["This is a test passage with some words and punctuation!", "Another sample text for LLM identification."]
feats = feat_extractor.transform(sample_texts)
print("RichStyleFeatures output shape:", feats.shape)

