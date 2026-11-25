"""Metrics checked against values worked out by hand (the arithmetic is in the comments)."""
import math

import numpy as np
import pytest

from wands_search import metrics as M

# Judgements for one query: a, d, e are Exact (2), c is Partial (1), b is Irrelevant (0).
J = {"a": 2, "b": 0, "c": 1, "d": 2, "e": 2}
# The system returns a, x (never judged, so gain 0), c, d. It misses e.
RUN = ["a", "x", "c", "d"]


def test_dcg_by_hand():
    # 2/log2(2) + 0/log2(3) + 1/log2(4) + 2/log2(5) = 2 + 0 + 0.5 + 0.861353 = 3.361353
    assert M.dcg([2, 0, 1, 2], 4) == pytest.approx(2 + 0 + 0.5 + 2 / math.log2(5))
    assert M.dcg([2, 0, 1, 2], 4) == pytest.approx(3.361353, abs=1e-6)


def test_ndcg_by_hand():
    # Ideal order of ALL judged gains: [2, 2, 2, 1, 0]; IDCG@4 = 2 + 2/log2(3) + 2/2 + 1/log2(5)
    #   = 2 + 1.261860 + 1 + 0.430677 = 4.692537.  NDCG@4 = 3.361353 / 4.692537 = 0.716319
    assert M.ndcg_at_k(RUN, J, k=4) == pytest.approx(0.716319, abs=1e-6)
    # @1: DCG = 2, IDCG = 2 -> 1.0
    assert M.ndcg_at_k(RUN, J, k=1) == pytest.approx(1.0)


def test_ndcg_perfect_and_empty():
    assert M.ndcg_at_k(["a", "d", "e", "c", "b"], J, k=10) == pytest.approx(1.0)
    assert M.ndcg_at_k([], J, k=10) == 0.0
    # no positive judgement -> undefined, not 0 (so it is excluded from the mean)
    assert math.isnan(M.ndcg_at_k(["b"], {"b": 0}, k=10))


def test_ndcg_ignores_order_of_ties_and_uses_unretrieved_ideal():
    # Retrieving only Partial items can never reach 1 when Exact items exist.
    assert M.ndcg_at_k(["c"], J, k=1) == pytest.approx(1 / 2)


def test_average_precision_by_hand():
    # Exact items: a, d, e (R = 3). Hits at rank 1 (P = 1/1) and rank 4 (P = 2/4).
    # AP = (1 + 0.5) / 3 = 0.5
    assert M.average_precision(RUN, J) == pytest.approx(0.5)


def test_recall_and_rr_by_hand():
    assert M.recall_at_k(RUN, J, k=2) == pytest.approx(1 / 3)  # only a
    assert M.recall_at_k(RUN, J, k=50) == pytest.approx(2 / 3)  # a, d
    assert M.reciprocal_rank(RUN, J) == 1.0
    assert M.reciprocal_rank(["x", "c", "d"], J) == pytest.approx(1 / 3)
    assert M.reciprocal_rank(["x", "c"], J) == 0.0
    # no Exact product for the query -> undefined
    assert math.isnan(M.reciprocal_rank(["c"], {"c": 1}))
    assert math.isnan(M.average_precision(["c"], {"c": 1}))


def test_judged_at_k():
    assert M.judged_at_k(RUN, J, k=4) == pytest.approx(3 / 4)  # x is unjudged


def test_evaluate_and_summarize_skip_undefined():
    qrels = {1: J, 2: {"c": 1}}
    run = {1: RUN, 2: ["c"]}
    pq = M.evaluate(run, qrels, [1, 2])
    s = M.summarize(pq)
    # query 2 has no Exact item: it counts for ndcg but not for mrr
    assert s["n_ndcg@10"] == 2 and s["n_mrr"] == 1
    assert s["mrr"] == pytest.approx(1.0)
    assert pq.loc[2, "ndcg@10"] == pytest.approx(1.0)
    assert np.isnan(pq.loc[2, "map"])
