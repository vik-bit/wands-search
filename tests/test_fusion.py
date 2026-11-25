import pytest

from wands_search import fusion


def test_rrf_by_hand():
    a = ["x", "y", "z"]
    b = ["y", "w"]
    ids, scores = fusion.rrf([a, b], k=60)
    # y: 1/62 + 1/61 ; x: 1/61 ; w: 1/62 ; z: 1/63
    assert ids == ["y", "x", "w", "z"]
    assert scores[0] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[3] == pytest.approx(1 / 63)


def test_rrf_weights_and_depth():
    ids, _ = fusion.rrf([["x"], ["y"]], k=60, weights=[1.0, 2.0])
    assert ids == ["y", "x"]
    ids, _ = fusion.rrf([["a", "b", "c"]], depth=2)
    assert ids == ["a", "b"]


def test_rrf_ties_are_deterministic():
    ids, _ = fusion.rrf([["b"], ["a"]])
    assert ids == ["a", "b"]


def test_minmax():
    assert fusion.minmax([3.0, 1.0, 2.0]) == [1.0, 0.0, 0.5]
    assert fusion.minmax([5.0, 5.0]) == [1.0, 1.0]
    assert fusion.minmax([]) == []


def test_weighted_by_hand():
    # system 1 scores x=10, y=0 -> 1, 0 ; system 2 scores y=0.9, z=0.7 -> 1, 0
    ids, scores = fusion.weighted([["x", "y"], ["y", "z"]], [[10, 0], [0.9, 0.7]], [0.3, 0.7])
    # x = 0.3, y = 0 + 0.7, z = 0
    assert ids == ["y", "x", "z"]
    assert scores == pytest.approx([0.7, 0.3, 0.0])
