"""Combining two or more ranked lists for one query.

  rrf       Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum_s 1 / (k + rank_s(d)),
            ranks starting at 1. Uses ranks only, so BM25 scores (unbounded, query-dependent) and
            cosine similarities (around 0.6-0.9) never have to be put on one scale.
  weighted  score(d) = sum_s w_s * minmax_s(d): each system's scores min-max normalised over its
            own returned list for this query. A product missing from a list gets 0 from it, the
            same as that list's worst returned product.
"""
from __future__ import annotations


def rrf(rankings: list[list], k: float = 60.0, weights: list[float] | None = None, depth: int | None = None):
    weights = weights or [1.0] * len(rankings)
    scores: dict = {}
    for w, ranked in zip(weights, rankings):
        for r, d in enumerate(ranked, start=1):
            scores[d] = scores.get(d, 0.0) + w / (k + r)
    return _sorted(scores, depth)


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def weighted(rankings: list[list], scores: list[list[float]], weights: list[float], depth: int | None = None):
    out: dict = {}
    for w, ranked, sc in zip(weights, rankings, scores):
        for d, s in zip(ranked, minmax(list(sc))):
            out[d] = out.get(d, 0.0) + w * s
    return _sorted(out, depth)


def _sorted(scores: dict, depth: int | None):
    # ties broken by product id so the order is deterministic
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if depth is not None:
        items = items[:depth]
    return [d for d, _ in items], [s for _, s in items]
