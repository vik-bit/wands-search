"""Tokenisation shared by every lexical component (BM25, features, query tags).

Lower-case, split on anything that is not a letter or digit, then an optional light plural
stemmer. Product search queries are short and mostly nouns ("barstools", "bar stool", "lamps"),
so plural folding matters more than full Porter stemming, which also conflates words that should
stay apart in a catalogue (Porter maps both "organic" and "organ" to "organ").
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_KEEP = {"glass", "dress", "mattress", "chaise", "this", "always", "series", "species",
         "canvas", "bus", "gas", "plus", "lotus", "cactus"}
_IRREGULAR = {"shelves": "shelf", "knives": "knife", "leaves": "leaf", "wolves": "wolf"}


def stem(word: str) -> str:
    if len(word) <= 3 or word.isdigit() or word in _KEEP:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"           # "accessories" -> "accessory"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]                 # "benches" -> "bench", "boxes" -> "box"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]                 # "lamps" -> "lamp"
    return word


def tokenize(text: str, do_stem: bool = True) -> list[str]:
    toks = _TOKEN.findall(str(text).lower())
    return [stem(t) for t in toks] if do_stem else toks
