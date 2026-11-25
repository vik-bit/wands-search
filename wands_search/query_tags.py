"""Rule-based query types for error analysis.

The word lists are short and hand-written. They are only used to group the test queries for
analysis, never as a ranking feature, so a missed colour word only moves a query between groups.
"""
from __future__ import annotations

import re

from .text import tokenize

COLOURS = {"white", "black", "grey", "gray", "blue", "navy", "green", "red", "pink", "yellow",
           "orange", "purple", "brown", "beige", "cream", "ivory", "gold", "silver", "teal",
           "turquoise", "tan", "copper", "bronze", "brass", "chrome", "charcoal", "aqua", "blush",
           "emerald", "mustard", "sage", "taupe", "walnut", "espresso", "natural", "clear", "multi"}
MATERIALS = {"wood", "wooden", "oak", "pine", "teak", "acacia", "bamboo", "metal", "iron", "steel",
             "aluminum", "glass", "marble", "stone", "granite", "concrete", "leather", "velvet",
             "linen", "cotton", "wool", "jute", "rattan", "wicker", "fabric", "plastic", "resin",
             "ceramic", "porcelain", "faux", "upholstered", "acrylic", "mirrored", "cedar", "maple",
             "cherry", "mahogany", "brass", "copper", "dolomite", "quartz", "silk", "sherpa", "foam"}
STYLES = {"modern", "contemporary", "farmhouse", "rustic", "industrial", "vintage", "traditional",
          "boho", "bohemian", "coastal", "mid", "century", "scandinavian", "minimalist", "glam",
          "shabby", "chic", "victorian", "french", "country", "retro", "art", "deco", "unique",
          "cute", "luxury", "elegant", "antique", "classic", "transitional", "southwestern"}
SIZES = {"small", "large", "big", "tall", "short", "mini", "oversized", "queen", "king", "twin",
         "full", "narrow", "wide", "long", "kids", "toddler", "xl", "compact", "round", "square"}


def tags(query: str, rare_vocab: set | None = None) -> dict:
    raw = query.lower().split()
    toks = set(tokenize(query, do_stem=False))
    n = len(raw)
    out = {
        "len_1_2": n <= 2, "len_3_4": 3 <= n <= 4, "len_5_plus": n >= 5,
        "colour": bool(toks & COLOURS), "material": bool(toks & MATERIALS),
        "style": bool(toks & STYLES), "size_or_number": bool(toks & SIZES) or bool(re.search(r"\d", query)),
    }
    out["any_attribute"] = out["colour"] or out["material"] or out["style"] or out["size_or_number"]
    # A word that (almost) never appears in product names/classes is usually a brand or product
    # line name ("guilford", "attleboro") or a misspelling.
    out["rare_word"] = bool(rare_vocab is not None and toks - rare_vocab)
    out["plain_type"] = not out["any_attribute"] and not out["rare_word"]
    return out
