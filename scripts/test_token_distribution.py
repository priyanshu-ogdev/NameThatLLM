import pandas as pd
import numpy as np

# Load train dataset
df = pd.read_parquet("data/processed/clean_corpus_long.parquet")
print("Total rows:", len(df))

# Approximate token count: rough estimate 1 word ~ 1.3 tokens, or split on whitespace
words = df["text"].apply(lambda s: len(str(s).split()))
chars = df["text"].apply(lambda s: len(str(s)))

print("Word count quantiles:")
print(words.quantile([0.5, 0.9, 0.95, 0.99, 1.0]))

print("\nChar count quantiles:")
print(chars.quantile([0.5, 0.9, 0.95, 0.99, 1.0]))

# Check how many texts have chars > 1200 (~288 tokens)
print("\nTexts with char len > 1150 (~288 tokens):", (chars > 1150).mean() * 100, "%")
print("Texts with char len > 1500 (~380 tokens):", (chars > 1500).mean() * 100, "%")
print("Texts with char len > 2000 (~500 tokens):", (chars > 2000).mean() * 100, "%")

