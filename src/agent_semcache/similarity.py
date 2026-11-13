"""Deterministic, local, lexical similarity between two prompts. No embeddings, no network.

This measures how many characters / words two prompts share. It does NOT measure meaning:
"capital of France" and "capital of Spain" overlap a lot, "automobile" and "car" overlap not at all.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Callable, Dict

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """NFKC, casefold, every run of punctuation / whitespace becomes one space, ends trimmed.

    A prompt with no letters or digits at all (for example "?!") falls back to its stripped, casefolded
    text so that two different punctuation-only prompts do not both normalize to "" and match each other.
    """
    if not isinstance(text, str):
        raise TypeError(f"prompt must be str, got {type(text).__name__}")
    folded = unicodedata.normalize("NFKC", text).casefold()
    out = _NON_WORD.sub(" ", folded).strip()
    return out if out else folded.strip()


def tokens(norm: str) -> list:
    return norm.split()


def char_ngrams(norm: str, n: int = 3) -> Counter:
    padded = f" {norm} "
    if len(padded) < n:
        return Counter([padded])
    return Counter(padded[i:i + n] for i in range(len(padded) - n + 1))


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def _cosine(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


MEASURES: Dict[str, Callable[[str, str], float]] = {
    "char3-jaccard": lambda x, y: _jaccard(char_ngrams(x), char_ngrams(y)),
    "char3-cosine": lambda x, y: _cosine(char_ngrams(x), char_ngrams(y)),
    "token-jaccard": lambda x, y: _jaccard(tokens(x), tokens(y)),
    "token-cosine": lambda x, y: _cosine(Counter(tokens(x)), Counter(tokens(y))),
}
DEFAULT_MEASURE = "char3-jaccard"


def similarity_norm(norm_a: str, norm_b: str, measure: str = DEFAULT_MEASURE) -> float:
    """Score two already-normalized strings, 0.0 to 1.0. Identical strings always score exactly 1.0."""
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}; choose one of {sorted(MEASURES)}")
    if norm_a == norm_b:
        return 1.0
    return round(MEASURES[measure](norm_a, norm_b), 12)


def similarity(a: str, b: str, measure: str = DEFAULT_MEASURE) -> float:
    return similarity_norm(normalize(a), normalize(b), measure)
