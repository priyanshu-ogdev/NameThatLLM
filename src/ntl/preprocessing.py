"""Robust, leak-free text preprocessing and stylometric feature extraction for Name That LLM."""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


def clean_text(text: str) -> str:
    """
    Carefully clean passage text without stripping stylometric signals.
    
    1. Preserves exact punctuation (em-dashes, quotes, ellipses, question marks).
    2. Preserves letter casing and capitalization cadences.
    3. Preserves markdown formatting characters (#, *, `, -).
    4. Normalizes unicode whitespace irregularities (non-breaking spaces, zero-width spaces).
    5. Normalizes line-endings to standard UNIX newlines.
    """
    if not isinstance(text, str):
        return ""
    # Normalize carriage returns
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces and zero-width spaces with standard spaces
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Remove control characters except tab and newline
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    # Strip external boundary whitespace
    return s.strip()


class RichStylometricExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 80+ interpretable stylometric linguistic and typographical features.
    
    Categories:
    - Length & dispersion (log word count, log char count, sentence length stats)
    - Typographical habits (em-dash vs en-dash vs hyphen, curly vs straight quotes)
    - Punctuation density (period, comma, colon, semicolon, exclamation, question mark, ellipsis)
    - Structural markers (markdown headers, backticks, bullet items, uppercase/digit density)
    - Invariant grammatical words (discourse transitions, modal verbs, pronouns, contractions)
    """

    PUNCTUATION_SYMBOLS = [
        "—", "–", "-", ",", ".", ";", ":", "!", "?", "…", "...",
        '"', "'", "’", "“", "”", "(", ")", "[", "]", "*", "#", "\n", "\t", "/", "%", "`"
    ]

    FUNCTION_WORDS = (
        "the a an and or but if then so because although however therefore thus "
        "moreover furthermore indeed perhaps might could would should may must can will "
        "we you i it they this that these those not no very more most such"
    ).split()

    CONTRACTION_REGEX = re.compile(r"\b\w+['’]\w+\b")
    WORD_REGEX = re.compile(r"\b\w+(?:['’]\w+)?\b")
    SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+\s*")

    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        # Determine feature names dynamically
        sample_feat = self._extract_single("Sample text. Indeed, we are testing!")
        self.feature_names_ = list(sample_feat.keys())
        return self

    def _extract_single(self, text: str) -> dict:
        text = clean_text(text)
        nc = max(len(text), 1)
        words = self.WORD_REGEX.findall(text)
        nw = max(len(words), 1)
        lower = [w.lower() for w in words]

        word_lengths = [len(w) for w in words] or [0]
        sentences = [s for s in self.SENTENCE_SPLIT_REGEX.split(text) if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences] or [0]

        feat = {
            "log_char_count": np.log1p(nc),
            "log_word_count": np.log1p(nw),
            "log_sentence_count": np.log1p(len(sentences)),
            "mean_word_length": float(np.mean(word_lengths)),
            "std_word_length": float(np.std(word_lengths)),
            "median_word_length": float(np.median(word_lengths)),
            "mean_sentence_length": float(np.mean(sentence_lengths)),
            "std_sentence_length": float(np.std(sentence_lengths)),
            "type_token_ratio": len(set(lower)) / nw,
            "short_word_ratio": sum(len(w) <= 3 for w in words) / nw,
            "long_word_ratio": sum(len(w) >= 8 for w in words) / nw,
            "uppercase_char_ratio": sum(c.isupper() for c in text) / nc,
            "digit_char_ratio": sum(c.isdigit() for c in text) / nc,
            "non_ascii_char_ratio": sum(ord(c) > 127 for c in text) / nc,
            "contraction_ratio": len(self.CONTRACTION_REGEX.findall(text)) / nw,
        }

        # Punctuation counts and frequencies
        for p in self.PUNCTUATION_SYMBOLS:
            cnt = text.count(p)
            safe_p = p.replace("\n", "\\n").replace("\t", "\\t")
            feat[f"punct_count_{safe_p}"] = float(cnt)
            feat[f"punct_freq_{safe_p}"] = float(cnt) / nw

        # Grammatical function word frequencies
        for fw in self.FUNCTION_WORDS:
            feat[f"func_word_{fw}"] = lower.count(fw) / nw

        return feat

    def transform(self, X):
        records = [list(self._extract_single(t).values()) for t in X]
        return np.asarray(records, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_ is None:
            sample_feat = self._extract_single("Sample text.")
            self.feature_names_ = list(sample_feat.keys())
        return np.array(self.feature_names_, dtype=object)


def build_multiview_preprocessor(style_weight: float = 0.2):
    """
    Builds the production-grade Multiview FeatureUnion pipeline:
    1. Full Character N-Grams (1-5 chars, 180k max features)
    2. Whitespace Word N-Grams (1-2 words, 150k max features, preserving attached punctuation)
    3. Word-Boundary Character N-Grams ('char_wb', 3-5 chars, 100k max features)
    4. Punctuation & Word Tokenizer (1-2 tokens, 100k max features)
    5. Scaled Rich Stylometrics (80+ linguistic features, scaled by StandardScaler, weighted by style_weight)
    """
    style_branch = Pipeline([
        ("extract", RichStylometricExtractor()),
        ("scale", StandardScaler())
    ])

    char_branch = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 5),
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
        max_features=180_000
    )

    word_branch = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=False,
        token_pattern=r"\S+",
        sublinear_tf=True,
        min_df=2,
        max_features=150_000
    )

    boundary_branch = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        max_features=100_000
    )

    punct_branch = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        ngram_range=(1, 2),
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
        max_features=100_000
    )

    return FeatureUnion(
        transformer_list=[
            ("char", char_branch),
            ("word", word_branch),
            ("boundary", boundary_branch),
            ("punct", punct_branch),
            ("style", style_branch)
        ],
        transformer_weights={"style": style_weight},
        n_jobs=1
    )

