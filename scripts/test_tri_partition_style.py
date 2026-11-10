import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

class RichStyleFeatures(BaseEstimator, TransformerMixin):
    PUNCTUATION = [
        "\u2014", "\u2013", "-", ",", ".", ";", ":", "!", "?", "\u2026", "...",
        '"', "'", "\u2018", "\u2019", "\u201c", "\u201d", "(", ")", "[", "]",
        "*", "#", "\n", "\t", "/", "%", "`",
    ]
    FUNCTION_WORDS = (
        "the a an and or but if then so because although however therefore thus "
        "moreover furthermore indeed perhaps might could would should may must can will "
        "we you i it they this that these those not no very more most such"
    ).split()
    CONTRACTION_RE = re.compile(r"\b\w+['\u2019\u2018]\w+\b")
    WORD_RE = re.compile(r"\b\w+(?:['\u2019\u2018]\w+)?\b")
    SENT_RE = re.compile(r"[.!?]+\s*")
    H_RE = re.compile(r"^\s*#{1,6}\s")
    LIST_RE = re.compile(r"^\s*[-*\u2022]\s|^\s*\d+\.\s")
    BOLD_RE = re.compile(r"^\s*\*\*")
    OPENING_RE = re.compile(r"^(certainly|sure|here is|as an|to answer|i would)", re.I)
    CLOSING_RE = re.compile(r"(overall|in conclusion|in summary|ultimately|hope this helps|to summarize)", re.I)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            t = text.strip() if isinstance(text, str) else ""
            nc = max(len(t), 1)
            words = self.WORD_RE.findall(t)
            nw = max(len(words), 1)
            lower = [w.lower() for w in words]
            w_lengths = [len(w) for w in words] or [0]
            sents = [s for s in self.SENT_RE.split(t) if s.strip()]
            s_lengths = [len(s.split()) for s in sents] or [0]

            feat = [
                np.log1p(nc), np.log1p(nw), np.log1p(len(sents)),
                float(np.mean(w_lengths)), float(np.std(w_lengths)), float(np.median(w_lengths)),
                float(np.mean(s_lengths)), float(np.std(s_lengths)),
                len(set(lower)) / nw,
                sum(len(w) <= 3 for w in words) / nw,
                sum(len(w) >= 8 for w in words) / nw,
                sum(c.isupper() for c in t) / nc,
                sum(c.isdigit() for c in t) / nc,
                sum(ord(c) > 127 for c in t) / nc,
                len(self.CONTRACTION_RE.findall(t)) / nw,
            ]
            for p in self.PUNCTUATION:
                cnt = t.count(p)
                feat.append(np.log1p(cnt))
                feat.append(cnt / nw)
            w_counts = Counter(lower)
            for fw in self.FUNCTION_WORDS:
                feat.append(w_counts[fw] / nw)

            # Positional Tri-Partition Stylometrics (Head 25%, Body 50%, Tail 25%)
            n_quarter = max(nw // 4, 1)
            head_words = words[:n_quarter]
            body_words = words[n_quarter:max(nw - n_quarter, n_quarter + 1)]
            tail_words = words[max(nw - n_quarter, 0):]

            head_text = " ".join(head_words)
            tail_text = " ".join(tail_words)

            head_w_lens = [len(w) for w in head_words] or [0]
            tail_w_lens = [len(w) for w in tail_words] or [0]
            body_w_lens = [len(w) for w in body_words] or [0]

            feat.extend([
                float(bool(self.H_RE.search(t))),
                float(bool(self.LIST_RE.search(t))),
                float(bool(self.BOLD_RE.search(t))),
                float(bool(self.OPENING_RE.search(head_text))),
                float(bool(self.CLOSING_RE.search(tail_text))),
                float(t.endswith("?")),
                float(t.endswith("!")),
                # Tri-partition word length moments
                float(np.mean(head_w_lens)),
                float(np.std(head_w_lens)),
                sum(c.isupper() for c in head_text) / max(len(head_text), 1),
                float(np.mean(tail_w_lens)),
                float(np.std(tail_w_lens)),
                sum(c.isupper() for c in tail_text) / max(len(tail_text), 1),
                tail_text.count(".") / max(len(tail_words), 1),
                tail_text.count("!") / max(len(tail_words), 1),
                tail_text.count("?") / max(len(tail_words), 1),
                float(np.mean(body_w_lens)),
                float(np.std(body_w_lens)),
            ])

            # Paragraph and Line Cadence
            paragraphs = [p for p in t.split("\n\n") if p.strip()]
            n_paras = max(len(paragraphs), 1)
            para_lengths = [len(p.split()) for p in paragraphs] or [0]
            lines = [l for l in t.split("\n") if l.strip()]
            line_lengths = [len(l.split()) for l in lines] or [0]

            feat.extend([
                np.log1p(n_paras),
                float(np.mean(para_lengths)),
                float(np.std(para_lengths)),
                np.log1p(max(len(lines), 1)),
                float(np.mean(line_lengths)),
                float(np.std(line_lengths)),
                t.count("\n\n") / max(t.count("\n"), 1),
                len(set(lower)) / max(np.sqrt(nw), 1e-6),    # Root Type-Token Ratio
                sum(len(w) >= 12 for w in words) / nw,        # Multisyllabic density
                t.count("```") / nw,                           # Code block density
                t.count("`") / nw,                             # Inline code density
                t.count("$") / nw,                             # LaTeX math density
                t.count("\\") / nw,                            # TeX / escape density
                t.count(":") / max(t.count(";") + 1, 1),       # Colon-to-semicolon ratio
                (t.count('"') + t.count("\u201c") + t.count("\u201d")) / nw,  # Quotation density
            ])
            rows.append(feat)
        return np.asarray(rows, dtype=np.float64)

if __name__ == "__main__":
    X = [
        "Hello world! Certainly here is an explanation. **Bold** heading.\n\nIn conclusion, hope this helps!",
        "",  # empty edge case
        "SingleWord",
        "A " * 300,
    ]
    res = RichStyleFeatures().fit_transform(X)
    print("Shape:", res.shape)
    print("All finite:", np.all(np.isfinite(res)))
    print("Success!")

