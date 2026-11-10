with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace(
    "# --- 3. 115-DIMENSIONAL RICH STYLOMETRIC EXTRACTOR ---",
    "# --- 3. 132-DIMENSIONAL RICH STYLOMETRIC EXTRACTOR ---"
)
text = text.replace(
    '"""Linguistic and typographical features capturing authorial habits."""',
    '"""Linguistic and typographical features (132 features) capturing authorial habits."""'
)
text = text.replace(
    'CONTRACTION_RE = re.compile(r"\\b\\w+[\'\']\\w+\\b")',
    'CONTRACTION_RE = re.compile(r"\\b\\w+[\'’\\u2019]\\w+\\b")'
)
text = text.replace(
    'WORD_RE = re.compile(r"\\b\\w+(?:[\'\']\\w+)?\\b")',
    'WORD_RE = re.compile(r"\\b\\w+(?:[\'’\\u2019]\\w+)?\\b")'
)

with open("notebooks/kaggle_tri_ensemble.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated header and regexes successfully!")

