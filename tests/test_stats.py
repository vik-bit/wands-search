import numpy as np
import pytest

from wands_search import stats


def test_exact_randomization_small():
    # Differences [1, 1, 1]: of the 8 sign patterns, only all-plus and all-minus reach |mean| = 1.
    assert stats.randomization_test([0, 0, 0], [1, 1, 1]) == pytest.approx(2 / 8)
    # Differences [1, -1]: every pattern has |mean| >= 0, so p = 1.
    assert stats.randomization_test([0, 0], [1, -1]) == pytest.approx(1.0)


def test_randomization_detects_a_real_shift_and_not_noise():
    rng = np.random.default_rng(1)
    a = rng.uniform(size=200)
    assert stats.randomization_test(a, a + 0.05 + rng.normal(0, 0.05, 200)) < 0.001
    b = a + rng.normal(0, 0.05, 200)
    assert stats.randomization_test(a, b) > 0.01


def test_bootstrap_ci_contains_mean_and_is_degenerate_for_constants():
    m, lo, hi = stats.bootstrap_ci(np.full(50, 0.3))
    assert m == lo == hi == pytest.approx(0.3)
    x = np.random.default_rng(0).normal(0.5, 0.1, 400)
    m, lo, hi = stats.bootstrap_ci(x)
    assert lo < m < hi
    # roughly mean +- 1.96 * sd / sqrt(n) = 0.5 +- 0.0098
    assert hi - lo == pytest.approx(2 * 1.96 * 0.1 / 20, rel=0.15)


def test_compare_drops_nan_pairs_and_counts():
    a = [0.1, 0.5, np.nan, 0.2]
    b = [0.2, 0.5, 0.9, 0.1]
    r = stats.compare(a, b)
    assert r["n"] == 3 and r["wins"] == 1 and r["losses"] == 1 and r["ties"] == 1
    assert r["delta"] == pytest.approx(0.0)
