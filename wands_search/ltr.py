"""Features for learning to rank, one row per (query, candidate product).

Candidates are the hybrid run's top 100 for the query. Feature groups (see docs/LEARNING.md for
why each one is there):

  bm25      BM25 of the query against each field separately, and the tuned weighted total
  dense     cosine similarity between the bge-small query and product embeddings
  rank      the candidate's position in the hybrid list (1 = top)
  coverage  share of the query's tokens found in the product name / in any field
  class     class_vote: share of the hybrid top 20 that has this product's class (a cheap,
            query-specific guess at "which category is this query about"); class_in_query: share
            of the product-class tokens that the query contains
  reviews   log review count, log rating count, average rating, "no rating" flag
  query     query length in tokens; product name length
  ce        cross-encoder score (only in the variant that uses it)
  oracle    1 if the product class equals WANDS' annotated query_class. NOT available at serving
            time (a real system would have to predict the query class); used only as an upper
            bound on what a perfect query classifier could add.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .bm25 import FIELDS
from .text import tokenize

GROUPS = {
    "bm25": [f"bm25_{f}" for f in FIELDS] + ["bm25_total"],
    "dense": ["dense_cos"],
    "rank": ["hybrid_rank"],
    "coverage": ["cover_name", "cover_any"],
    "class": ["class_vote", "class_in_query"],
    "reviews": ["log_reviews", "log_ratings", "avg_rating", "rating_missing"],
    "query": ["query_len", "name_len"],
    "ce": ["ce_score"],
    "oracle": ["oracle_class_match"],
}


def build_features(qid: int, query: str, query_class: str, candidates: list[int], *,
                   bm25_index, bm25_cfg: dict, pid_to_row: dict, products: pd.DataFrame,
                   query_vec: np.ndarray, product_vecs: np.ndarray,
                   name_tokens: list[set], all_tokens: list[set], cls_tokens: list[set],
                   ce_scores: list[float] | None = None) -> pd.DataFrame:
    rows = [pid_to_row[p] for p in candidates]
    fs = bm25_index.field_scores(query, bm25_cfg["stem"])
    feats = {f"bm25_{f}": fs[f][rows] for f in FIELDS}
    feats["bm25_total"] = sum(bm25_cfg["weights"][f] * feats[f"bm25_{f}"] for f in FIELDS)
    feats["dense_cos"] = product_vecs[rows] @ query_vec
    feats["hybrid_rank"] = np.arange(1, len(rows) + 1, dtype=float)

    qt = set(tokenize(query))
    nq = max(len(qt), 1)
    feats["cover_name"] = np.array([len(qt & name_tokens[r]) / nq for r in rows])
    feats["cover_any"] = np.array([len(qt & all_tokens[r]) / nq for r in rows])

    classes = products["cls"].to_numpy()[rows]
    vote = Counter(classes[:20])
    top = min(20, len(rows)) or 1
    feats["class_vote"] = np.array([vote.get(c, 0) / top if c else 0.0 for c in classes])
    feats["class_in_query"] = np.array(
        [len(cls_tokens[r] & qt) / len(cls_tokens[r]) if cls_tokens[r] else 0.0 for r in rows])

    feats["log_reviews"] = np.log1p(products["review_count"].to_numpy()[rows])
    feats["log_ratings"] = np.log1p(products["rating_count"].to_numpy()[rows])
    feats["avg_rating"] = products["average_rating"].to_numpy()[rows]
    feats["rating_missing"] = products["rating_missing"].to_numpy()[rows].astype(float)
    feats["query_len"] = np.full(len(rows), float(len(tokenize(query))))
    feats["name_len"] = np.array([float(len(name_tokens[r])) for r in rows])
    feats["ce_score"] = np.asarray(ce_scores if ce_scores is not None else [np.nan] * len(rows), dtype=float)
    feats["oracle_class_match"] = np.array([float(bool(c) and c == query_class) for c in classes])

    df = pd.DataFrame(feats)
    df.insert(0, "product_id", candidates)
    df.insert(0, "query_id", qid)
    return df


def token_sets(products: pd.DataFrame):
    from .bm25 import FIELDS as F
    name = [set(tokenize(t)) for t in products["name"]]
    allf = [set(tokenize(" ".join(str(products[f].iat[i]) for f in F))) for i in range(len(products))]
    cls = [set(tokenize(t)) for t in products["cls"]]
    return name, allf, cls


def columns(groups) -> list[str]:
    return [c for g in groups for c in GROUPS[g]]
