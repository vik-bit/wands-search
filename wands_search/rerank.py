"""Cross-encoder reranking of a first-stage candidate list.

A bi-encoder (the dense retriever) embeds the query and the product separately and compares two
vectors, so it can be precomputed for 43k products. A cross-encoder reads the query and the
product text *together* through one transformer and outputs one relevance score. It sees
interactions such as "the query says 'outdoor' and this product says 'indoor'", but it costs one
forward pass per (query, product) pair, so it only runs on the top k of a cheaper first stage.

Model: ms-marco-MiniLM-L-6-v2 (22M parameters), trained on MS MARCO web passages, not on
product data: this measures how far a general-purpose reranker transfers to furniture search.
"""
from __future__ import annotations

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class CrossEncoder:
    def __init__(self, model: str = MODEL, threads: int | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model = TextCrossEncoder(model_name=model, threads=threads)

    def score(self, query: str, texts: list[str], batch_size: int = 64) -> list[float]:
        return [float(s) for s in self.model.rerank(query, texts, batch_size=batch_size)]


def rerank(candidates: list, scores: list[float], depth: int):
    """Reorder the first `depth` candidates by `scores` (one per candidate in that prefix);
    the rest keep their first-stage order below them."""
    head = sorted(zip(candidates[:depth], scores[:depth]), key=lambda x: -x[1])
    return [d for d, _ in head] + list(candidates[depth:])
