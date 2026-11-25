"""Paired comparisons between two systems evaluated on the same queries.

Queries are the unit of sampling: the question is "would system B still beat A on a different
sample of queries like these?", so we resample queries, keeping each query's pair of scores
together (paired). Pairing removes the large query-to-query variance (some queries are easy for
every system), which is what makes small differences detectable on ~150 queries.

  bootstrap_ci      percentile CI of a mean, resampling queries with replacement.
  paired_bootstrap  CI of mean(B - A) over the same resamples.
  randomization     two-sided paired randomization (sign-flip) test of mean(B - A) = 0. Under the
                    null the two systems are exchangeable per query, so each difference's sign is
                    flipped at random. Exact enumeration when there are few queries.
Queries where either system's metric is NaN (undefined) are dropped before comparing.
"""
from __future__ import annotations

import itertools

import numpy as np


def _clean(a, b=None):
    a = np.asarray(a, dtype=float)
    if b is None:
        return a[~np.isnan(a)]
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    x = _clean(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Mean of (b - a) and its percentile CI."""
    a, b = _clean(a, b)
    return bootstrap_ci(b - a, n_boot=n_boot, alpha=alpha, seed=seed)


def randomization_test(a, b, n_perm: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for mean(b - a) = 0 by random sign flips of the paired differences."""
    a, b = _clean(a, b)
    d = b - a
    n = len(d)
    observed = abs(d.mean())
    if n <= 16:  # enumerate all 2^n sign patterns exactly
        signs = np.array(list(itertools.product((1, -1), repeat=n)), dtype=float)
        stats = np.abs((signs * d).mean(axis=1))
        return float(np.mean(stats >= observed - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice((1.0, -1.0), size=(n_perm, n))
    stats = np.abs((signs * d).mean(axis=1))
    # +1 smoothing: the observed assignment is itself one of the possible sign patterns
    return float((np.sum(stats >= observed - 1e-12) + 1) / (n_perm + 1))


def compare(a, b, **kw) -> dict:
    a, b = _clean(a, b)
    mean, lo, hi = paired_bootstrap(a, b, **{k: v for k, v in kw.items() if k in ("n_boot", "seed")})
    d = b - a
    return {"delta": mean, "ci_low": lo, "ci_high": hi,
            "p_value": randomization_test(a, b, seed=kw.get("seed", 0)),
            "wins": int(np.sum(d > 1e-9)), "losses": int(np.sum(d < -1e-9)),
            "ties": int(np.sum(np.abs(d) <= 1e-9)), "n": int(len(d))}
