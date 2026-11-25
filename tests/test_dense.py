"""Search-side tests on random vectors (no model download needed)."""
import numpy as np
import pytest

from wands_search.dense import ExactIndex, overlap_at_k, product_text


def _data(n=2000, d=32, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(20, d))
    x = centres[rng.integers(0, 20, n)] + 0.3 * rng.normal(size=(n, d))
    q = centres[rng.integers(0, 20, 50)] + 0.3 * rng.normal(size=(50, d))
    return x.astype(np.float32), q.astype(np.float32)


def test_exact_index_matches_brute_force():
    x, q = _data()
    idx, sim = ExactIndex(x).search(q, 10)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    ref = np.argsort(-(qn @ xn.T), axis=1)[:, :10]
    assert overlap_at_k(idx, ref, 10) == 1.0
    assert np.all(np.diff(sim, axis=1) <= 1e-6)  # sorted best first


def test_vecdex_close_to_exact():
    pytest.importorskip("vecdex")
    from wands_search.dense import VecdexIndex
    x, q = _data()
    exact, _ = ExactIndex(x).search(q, 10)
    approx, sim = VecdexIndex(x, ef=100).search(q, 10)
    assert overlap_at_k(approx, exact, 10) > 0.95
    assert np.all(sim <= 1.0 + 1e-5)


def test_product_text_skips_empty_parts():
    row = {"name": "oak table", "cls": "", "attrs": "white oak " * 40}
    t = product_text(row)
    assert t.startswith("oak table | white oak")
    assert len(t.split("|")[1].split()) == 30
