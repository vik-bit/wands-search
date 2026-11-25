"""BM25 over several product fields, combined as a weighted sum of per-field BM25 scores.

For each field we precompute the BM25 weight of every (product, term) pair once:

    w(d, t) = idf(t) * tf(d,t) * (k1 + 1) / (tf(d,t) + k1 * (1 - b + b * len(d) / avglen))
    idf(t)  = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))           (Lucene's non-negative idf)

and store it as a sparse matrix, so scoring a query is a sum of a few matrix columns. The field
score is then  score(d) = sum_f weight_f * BM25_f(q, d).

This is a weighted sum of separate BM25 scores, not BM25F. BM25F mixes the weighted term
frequencies across fields *before* the tf saturation, which is the principled version; the sum is
simpler, gives one BM25 score per field that LambdaMART can also use as a feature, and is what
many engines' multi-field query (e.g. Elasticsearch `most_fields`) does in practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .text import tokenize

FIELDS = ("name", "cls", "hierarchy", "attrs", "description")


@dataclass
class FieldIndex:
    weights: sp.csc_matrix  # n_docs x vocab, BM25 term weights


@dataclass
class BM25Index:
    vocab: dict
    fields: dict = field(default_factory=dict)
    n_docs: int = 0

    @classmethod
    def build(cls, products, fields=FIELDS, k1: float = 1.2, b: float = 0.75, do_stem: bool = True):
        """`products` is a DataFrame with one text column per field; row order = doc index."""
        return cls.from_tokens(tokenise_fields(products, fields, do_stem), k1=k1, b=b)

    @classmethod
    def from_tokens(cls, tokenised: dict, k1: float = 1.2, b: float = 0.75):
        """Build from {field: [token list per product]} (tokenise once, try many k1 / b)."""
        fields = list(tokenised)
        n_docs = len(next(iter(tokenised.values())))
        vocab: dict[str, int] = {}
        for f in fields:
            for toks in tokenised[f]:
                for t in toks:
                    vocab.setdefault(t, len(vocab))
        idx = cls(vocab=vocab, n_docs=n_docs)
        for f in fields:
            idx.fields[f] = FieldIndex(_bm25_matrix(tokenised[f], vocab, k1, b))
        return idx

    def field_scores(self, query: str, do_stem: bool = True) -> dict[str, np.ndarray]:
        """BM25 score of every product for the query, per field."""
        cols = [self.vocab[t] for t in tokenize(query, do_stem) if t in self.vocab]
        out = {}
        for f, fi in self.fields.items():
            if cols:
                out[f] = np.asarray(fi.weights[:, cols].sum(axis=1)).ravel()
            else:
                out[f] = np.zeros(self.n_docs)
        return out

    def score(self, query: str, weights: dict, do_stem: bool = True) -> np.ndarray:
        fs = self.field_scores(query, do_stem)
        return sum(weights.get(f, 0.0) * s for f, s in fs.items())


def tokenise_fields(products, fields=FIELDS, do_stem: bool = True) -> dict:
    return {f: [tokenize(t, do_stem) for t in products[f]] for f in fields}


def _bm25_matrix(docs: list[list[str]], vocab: dict, k1: float, b: float) -> sp.csc_matrix:
    rows, cols, tfs = [], [], []
    lengths = np.zeros(len(docs))
    for d, toks in enumerate(docs):
        lengths[d] = len(toks)
        counts: dict[int, int] = {}
        for t in toks:
            j = vocab[t]
            counts[j] = counts.get(j, 0) + 1
        rows.extend([d] * len(counts))
        cols.extend(counts.keys())
        tfs.extend(counts.values())
    rows, cols, tf = np.array(rows), np.array(cols), np.array(tfs, dtype=float)
    n = len(docs)
    avg = lengths.mean() if lengths.mean() > 0 else 1.0
    df = np.bincount(cols, minlength=len(vocab)).astype(float)
    idf = np.log1p((n - df + 0.5) / (df + 0.5))
    norm = k1 * (1 - b + b * lengths[rows] / avg)
    w = idf[cols] * tf * (k1 + 1) / (tf + norm)
    return sp.csr_matrix((w, (rows, cols)), shape=(n, len(vocab))).tocsc()


def top_k(scores: np.ndarray, k: int, ids=None):
    """Indices (or ids) of the k highest scores, best first. Zero scores are dropped."""
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="stable")]
    order = order[scores[order] > 0]
    return order if ids is None else [ids[i] for i in order]
