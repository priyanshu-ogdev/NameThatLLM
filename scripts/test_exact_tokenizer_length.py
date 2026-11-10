import pandas as pd
import numpy as np
from transformers import AutoTokenizer

df = pd.read_parquet("data/processed/clean_corpus_long.parquet")
texts = df["text"].dropna().sample(2000, random_state=42).tolist()

# Let's test with a standard BPE/WordPiece tokenizer (e.g. bert-base-uncased or gpt2 or similar local if available)
# Or simple regex based subword estimator
import re
def rough_bpe_tokens(t):
    # Splits on punctuation, whitespace, and subwords
    pieces = re.findall(r"\w+|[^\w\s]", str(t))
    return len(pieces)

lengths = [rough_bpe_tokens(t) for t in texts]
lengths = pd.Series(lengths)
print("Subword token quantiles across 2,000 real passages:")
print("50%:", lengths.quantile(0.50))
print("75%:", lengths.quantile(0.75))
print("90%:", lengths.quantile(0.90))
print("95%:", lengths.quantile(0.95))
print("99%:", lengths.quantile(0.99))
print("Max:", lengths.max())
print("\nPercentage of texts truncated if max_length=288:", (lengths > 288).mean() * 100, "%")
print("Percentage of texts truncated if max_length=352:", (lengths > 352).mean() * 100, "%")
print("Percentage of texts truncated if max_length=384:", (lengths > 384).mean() * 100, "%")
print("Percentage of texts truncated if max_length=400:", (lengths > 400).mean() * 100, "%")

