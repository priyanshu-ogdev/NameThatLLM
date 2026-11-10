import sys
sys.path.insert(0, ".")
from notebooks.kaggle_tri_ensemble import RichStyleFeatures

rf = RichStyleFeatures()
texts = [
    "I don't know, but it’s clear they didn’t want to.",
    "Overall, in conclusion, this is a very interesting LLM generation with # heading and **bold**.",
    ""
]
feat = rf.transform(texts)
print("Shape:", feat.shape)
assert feat.shape == (3, 132), f"Expected (3, 132), got {feat.shape}"
print("RichStyleFeatures 132-dim test passed with 100% success!")
