"""Small shared text helpers (tokenising, stop-words, light stemming)."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her
    here hers him his how i if in into is it its itself just me more most my no nor not of off on once only or
    other our ours out over own same she should so some such than that the their theirs them then there these
    they this those through to too under until up very was we were what when where which while who whom why
    will with would you your yours tell me explain describe please much many""".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SUFFIXES = ("ingly", "edly", "ing", "ed", "s", "ly")


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stop-word-free, lightly stemmed tokens."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]
