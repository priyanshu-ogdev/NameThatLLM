import re

with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("\r\n", "\n")

# 1. Imports cleanup
old_imp = """import concurrent.futures
import gc
import hashlib
import json
import re
import time
import warnings
from pathlib import Path"""

new_imp = """from collections import Counter
import gc
import hashlib
import json
import re
import time
import warnings
from pathlib import Path"""

if old_imp in text:
    text = text.replace(old_imp, new_imp, 1)
    print("1. Cleaned up imports and added Counter.")
else:
    print("Could not find old_imp!")

# Remove unused CosineAnnealingLR
old_lr = """from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR"""
new_lr = """from torch.optim import AdamW"""
if old_lr in text:
    text = text.replace(old_lr, new_lr, 1)
    print("2. Removed unused CosineAnnealingLR.")
else:
    print("Could not find old_lr!")

# 2. Update docstring and regexes in RichStyleFeatures
old_style_hdr = """# --- 3. 115-DIMENSIONAL RICH STYLOMETRIC EXTRACTOR ---

class RichStyleFeatures(BaseEstimator, TransformerMixin):
    \"\"\"Linguistic and typographical features capturing authorial habits.\"\"\"
    PUNCTUATION = [
        "\u2014", "\u2013", "-", ",", ".", ";", ":", "!", "?", "\u2026", "...",
        '"', "'", "\u2018", "\u201c", "\u201d", "(", ")", "[", "]",
        "*", "#", "\n", "\t", "/", "%", "`",
    ]
    FUNCTION_WORDS = (
        "the a an and or but if then so because although however therefore thus "
        "moreover furthermore indeed perhaps might could would should may must can will "
        "we you i it they this that these those not no very more most such"
    ).split()
    CONTRACTION_RE = re.compile(r"\\b\\w+['']\\w+\\b")
    WORD_RE = re.compile(r"\\b\\w+(?:['']\\w+)?\\b")"""

new_style_hdr = """# --- 3. 132-DIMENSIONAL RICH STYLOMETRIC EXTRACTOR ---

class RichStyleFeatures(BaseEstimator, TransformerMixin):
    \"\"\"Linguistic and typographical features (132 features) capturing authorial habits.\"\"\"
    PUNCTUATION = [
        "\u2014", "\u2013", "-", ",", ".", ";", ":", "!", "?", "\u2026", "...",
        '"', "'", "\u2018", "\u201c", "\u201d", "(", ")", "[", "]",
        "*", "#", "\n", "\t", "/", "%", "`",
    ]
    FUNCTION_WORDS = (
        "the a an and or but if then so because although however therefore thus "
        "moreover furthermore indeed perhaps might could would should may must can will "
        "we you i it they this that these those not no very more most such"
    ).split()
    CONTRACTION_RE = re.compile(r"\\b\\w+['’\u2019]\\w+\\b")
    WORD_RE = re.compile(r"\\b\\w+(?:['’\u2019]\\w+)?\\b")"""

if old_style_hdr in text:
    text = text.replace(old_style_hdr, new_style_hdr, 1)
    print("3. Updated RichStyleFeatures header, feature count (132), and curly apostrophe regexes.")
else:
    print("Could not find old_style_hdr!")

# 3. Counter optimization in RichStyleFeatures
old_fw_loop = """            for fw in self.FUNCTION_WORDS:
                feat.append(lower.count(fw) / nw)"""

new_fw_loop = """            w_counts = Counter(lower)
            for fw in self.FUNCTION_WORDS:
                feat.append(w_counts[fw] / nw)"""

if old_fw_loop in text:
    text = text.replace(old_fw_loop, new_fw_loop, 1)
    print("4. Replaced lower.count loop with Counter in RichStyleFeatures.")
else:
    print("Could not find old_fw_loop!")

# 4. Iterative DSU find
old_dsu = """class DSU:
    \"\"\"Disjoint Set Union with path compression.\"\"\"
    def __init__(self, elements):
        self.p = {x: x for x in elements}

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[ry] = rx"""

new_dsu = """class DSU:
    \"\"\"Disjoint Set Union with iterative two-pass path compression (immune to recursion limits).\"\"\"
    def __init__(self, elements):
        self.p = {x: x for x in elements}

    def find(self, x):
        root = x
        while self.p[root] != root:
            root = self.p[root]
        curr = x
        while curr != root:
            nxt = self.p[curr]
            self.p[curr] = root
            curr = nxt
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[ry] = rx"""

if old_dsu in text:
    text = text.replace(old_dsu, new_dsu, 1)
    print("5. Upgraded DSU.find to iterative path compression.")
else:
    print("Could not find old_dsu!")

# 5. Upfront length check in validate_and_save_submission
old_val = """def validate_and_save_submission(test_ids, predictions, output_path):
    \"\"\"Atomic parquet write with schema, null, and integrity validation.\"\"\"
    sub = pd.DataFrame({"id": test_ids, "model": predictions})"""

new_val = """def validate_and_save_submission(test_ids, predictions, output_path):
    \"\"\"Atomic parquet write with schema, null, and integrity validation.\"\"\"
    if len(predictions) != len(test_ids):
        raise ValueError(f"Row count mismatch: expected {len(test_ids)} IDs, got {len(predictions)} predictions")
    sub = pd.DataFrame({"id": test_ids, "model": predictions})"""

if old_val in text:
    text = text.replace(old_val, new_val, 1)
    print("6. Added upfront length check to validate_and_save_submission.")
else:
    print("Could not find old_val!")

# 6. Remove redundant .clone() in best_state copies
old_clone_deb = "best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}"
new_clone_deb = "best_state = {k: v.cpu() for k, v in model.state_dict().items()}"
if old_clone_deb in text:
    text = text.replace(old_clone_deb, new_clone_deb, 1)
    print("7. Removed redundant clone() from transformer best_state.")

old_clone_qwen = """            best_lora_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items() if "lora_" in k}
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}"""
new_clone_qwen = """            best_lora_state = {k: v.cpu() for k, v in backbone.state_dict().items() if "lora_" in k}
            best_head_state = {k: v.cpu() for k, v in head.state_dict().items()}"""
if old_clone_qwen in text:
    text = text.replace(old_clone_qwen, new_clone_qwen, 1)
    print("8. Removed redundant clone() from Qwen best_state.")

with open("notebooks/kaggle_tri_ensemble.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Section 3 final refinements applied successfully!")

