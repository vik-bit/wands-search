"""Dense retrieval: a small open embedding model + vecdex (HNSW) or exact inner-product search.

Model: BAAI/bge-small-en-v1.5 (33M parameters, 384 dimensions) through fastembed's ONNX runtime,
so it runs on a laptop CPU. Queries go through `query_embed`, which adds bge's retrieval
instruction ("Represent this sentence for searching relevant passages: "); products do not.

Product text is name + class + attribute values, not the description. Descriptions are long
marketing text ("good, deep sleep can be quite difficult to have in this busy age ..."), which
costs most of the encoding time and pulls the embedding away from what the product *is*.
"""
from __future__ import annotations

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"


def product_text(row, max_attr_words: int = 30) -> str:
    attrs = " ".join(str(row["attrs"]).split()[:max_attr_words])
    parts = [str(row["name"]), str(row["cls"]), attrs]
    return " | ".join(p for p in parts if p)


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Encoder:
    def __init__(self, model: str = MODEL, threads: int | None = None):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model, threads=threads)

    def products(self, texts, batch_size: int = 256) -> np.ndarray:
        return _normalise(np.stack(list(self.model.embed(list(texts), batch_size=batch_size))))

    def queries(self, texts) -> np.ndarray:
        return _normalise(np.stack(list(self.model.query_embed(list(texts)))))


class ExactIndex:
    """Brute-force inner product over normalised vectors (= cosine). The reference for vecdex."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = _normalise(vectors)

    def search(self, queries: np.ndarray, k: int):
        sims = _normalise(queries) @ self.vectors.T
        k = min(k, sims.shape[1])
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(len(sims))[:, None]
        order = np.argsort(-sims[rows, part], axis=1, kind="stable")
        idx = part[rows, order]
        return idx, sims[rows, idx]


class VecdexIndex:
    """vecdex HNSW with the cosine metric. vecdex returns distance = 1 - cosine."""

    def __init__(self, vectors: np.ndarray, M: int = 16, ef_construction: int = 200, ef: int = 200):
        import vecdex
        vectors = _normalise(vectors)
        self.ef = ef
        self.index = vecdex.Index(dim=vectors.shape[1], max_elements=len(vectors), metric="cosine",
                                  M=M, ef_construction=ef_construction)
        self.index.add(vectors)

    def search(self, queries: np.ndarray, k: int, ef: int | None = None):
        labels, dist = self.index.search(_normalise(queries), k=k, ef=max(ef or self.ef, k))
        return np.asarray(labels, dtype=np.int64), 1.0 - np.asarray(dist, dtype=np.float32)


def overlap_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Mean share of the top k of `b` (reference) that also appears in the top k of `a`."""
    return float(np.mean([len(set(x[:k]) & set(y[:k])) / k for x, y in zip(a, b)]))
