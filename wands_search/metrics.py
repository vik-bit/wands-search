"""Ranking metrics with graded relevance, computed per query.

A run maps query_id -> ranked list of product_ids (best first).
qrels maps query_id -> {product_id: gain}, gains 2 (Exact), 1 (Partial), 0 (Irrelevant).

  ndcg@k    graded: DCG = sum_i gain_i / log2(i + 1), divided by the DCG of the ideal ordering of
            *all* judged products for the query (not just the retrieved ones). Linear gain.
  ap        binary, relevant = Exact. Mean of precision@i at each relevant position, divided by
            the number of Exact products for the query (trec_eval convention), over the run's depth.
  recall@k  binary, relevant = Exact: share of the query's Exact products found in the top k.
  rr        binary, relevant = Exact: 1 / rank of the first Exact product, 0 if none in the run.
  judged@k  share of the top k that have any judgement. Unjudged products count as gain 0.

A metric that is undefined for a query (no positive judgement at all for ndcg, no Exact product
for the binary metrics) is NaN for that query and left out of the mean. The count of queries
that each mean is taken over is reported alongside it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EXACT = 2


def dcg(gains, k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(list(gains)[:k]))


def ndcg_at_k(ranked, judgements: dict, k: int = 10) -> float:
    ideal = dcg(sorted(judgements.values(), reverse=True), k)
    if ideal == 0:
        return float("nan")
    return dcg((judgements.get(p, 0) for p in ranked), k) / ideal


def _n_relevant(judgements: dict, threshold: int) -> int:
    return sum(1 for g in judgements.values() if g >= threshold)


def average_precision(ranked, judgements: dict, threshold: int = EXACT) -> float:
    n_rel = _n_relevant(judgements, threshold)
    if n_rel == 0:
        return float("nan")
    hits, total = 0, 0.0
    for i, p in enumerate(ranked, start=1):
        if judgements.get(p, 0) >= threshold:
            hits += 1
            total += hits / i
    return total / n_rel


def recall_at_k(ranked, judgements: dict, k: int, threshold: int = EXACT) -> float:
    n_rel = _n_relevant(judgements, threshold)
    if n_rel == 0:
        return float("nan")
    return sum(1 for p in ranked[:k] if judgements.get(p, 0) >= threshold) / n_rel


def reciprocal_rank(ranked, judgements: dict, threshold: int = EXACT) -> float:
    if _n_relevant(judgements, threshold) == 0:
        return float("nan")
    for i, p in enumerate(ranked, start=1):
        if judgements.get(p, 0) >= threshold:
            return 1.0 / i
    return 0.0


def judged_at_k(ranked, judgements: dict, k: int = 10) -> float:
    top = ranked[:k]
    return sum(1 for p in top if p in judgements) / max(len(top), 1)


METRICS = ("ndcg@10", "map", "recall@50", "mrr", "judged@10")


def evaluate(run: dict, qrels: dict, query_ids) -> pd.DataFrame:
    """Per-query metrics for the given queries. A query missing from the run scores as empty."""
    rows = []
    for qid in query_ids:
        ranked = list(run.get(qid, []))
        j = qrels.get(qid, {})
        rows.append({"query_id": qid,
                     "ndcg@10": ndcg_at_k(ranked, j, 10),
                     "map": average_precision(ranked, j),
                     "recall@50": recall_at_k(ranked, j, 50),
                     "mrr": reciprocal_rank(ranked, j),
                     "judged@10": judged_at_k(ranked, j, 10)})
    return pd.DataFrame(rows).set_index("query_id")


def summarize(per_query: pd.DataFrame) -> dict:
    out = {}
    for m in per_query.columns:
        col = per_query[m].dropna()
        out[m] = float(col.mean()) if len(col) else float("nan")
        out[f"n_{m}"] = int(len(col))
    return out


def as_array(per_query: pd.DataFrame, metric: str, query_ids) -> np.ndarray:
    return per_query.loc[list(query_ids), metric].to_numpy(dtype=float)
