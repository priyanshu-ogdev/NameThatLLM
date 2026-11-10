import sys
from pathlib import Path

path = Path("notebooks/kaggle_tri_ensemble.py")
content = path.read_text(encoding="utf-8")

# 1. Add import scipy alongside rankdata
target_scipy = "from scipy.stats import rankdata"
repl_scipy = "import scipy\nfrom scipy.stats import rankdata"
assert target_scipy in content, "target_scipy not found"
content = content.replace(target_scipy, repl_scipy, 1)

# 2. Fix compute_passage_ranks
target_rankdata = "def compute_passage_ranks(margins: np.ndarray) -> np.ndarray:\n    \"\"\"Compute ordinal ranks per row [1, K]. Vectorized via scipy.stats.rankdata.\"\"\"\n    return scipy.stats.rankdata(margins, axis=1)"
repl_rankdata = "def compute_passage_ranks(margins: np.ndarray) -> np.ndarray:\n    \"\"\"Compute ordinal ranks per row [1, K]. Vectorized via scipy.stats.rankdata.\"\"\"\n    return rankdata(margins, axis=1)"
assert target_rankdata in content, "target_rankdata not found"
content = content.replace(target_rankdata, repl_rankdata, 1)

# 3. Clean up clean_text fast path
target_clean_text = """def clean_text(text: str) -> str:
    \"\"\"Normalize whitespace and line endings while preserving casing, punctuation, and markdown.\"\"\"
    if not isinstance(text, str):
        return ""
    if "\\r" not in text and "\\u00a0" not in text and "\\u200b" not in text and "\\ufeff" not in text:
        return text.strip()
    s = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    s = s.replace("\\u00a0", " ").replace("\\u200b", "").replace("\\ufeff", "")
    s = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]", "", s)
    return s.strip()"""

repl_clean_text = """def clean_text(text: str) -> str:
    \"\"\"Normalize whitespace and line endings while preserving casing, punctuation, and markdown.\"\"\"
    if not isinstance(text, str):
        return ""
    s = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    s = s.replace("\\u00a0", " ").replace("\\u200b", "").replace("\\ufeff", "")
    s = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]", "", s)
    return s.strip()"""

assert target_clean_text in content, "target_clean_text not found"
content = content.replace(target_clean_text, repl_clean_text, 1)

# 4. Protect against ZeroDivisionError in fold evaluation logging
target_div = "{np.sum(deb_mask)//(n_train//n_splits)}/{n_splits}"
repl_div = "{np.sum(deb_mask)//max(n_train//n_splits, 1)}/{n_splits}"
assert target_div in content, "target_div not found"
content = content.replace(target_div, repl_div)

target_bge_div = "{np.sum(bge_mask)//(n_train//n_splits)}/{n_splits}"
repl_bge_div = "{np.sum(bge_mask)//max(n_train//n_splits, 1)}/{n_splits}"
assert target_bge_div in content, "target_bge_div not found"
content = content.replace(target_bge_div, repl_bge_div)

target_qwen_div = "{np.sum(qwen_mask)//(n_train//n_splits)}/{n_splits}"
repl_qwen_div = "{np.sum(qwen_mask)//max(n_train//n_splits, 1)}/{n_splits}"
assert target_qwen_div in content, "target_qwen_div not found"
content = content.replace(target_qwen_div, repl_qwen_div)

path.write_text(content, encoding="utf-8")
print("Successfully applied forensic hardening to notebooks/kaggle_tri_ensemble.py!")

