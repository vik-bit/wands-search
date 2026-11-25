from pathlib import Path

from wands_search import data

TINY = Path(__file__).parent / "fixtures" / "tiny"


def test_products_fields_and_missing_values():
    p = data.load_products(TINY)
    assert list(p.product_id) == [0, 1, 2, 3]
    # attrs keeps colour/material/style/shape values and drops compliance + size fields
    assert p.loc[0, "attrs"] == "white solid oak"
    assert p.loc[2, "attrs"] == "metal black industrial"
    assert p.loc[3, "attrs"] == "round oak"
    # missing description / class / ratings become "" / 0 with a flag
    assert p.loc[1, "description"] == "" and p.loc[3, "cls"] == ""
    assert p.loc[1, "rating_missing"] == 1 and p.loc[1, "review_count"] == 0
    assert p.loc[0, "rating_missing"] == 0 and p.loc[0, "review_count"] == 10


def test_qrels_gains_and_duplicates_take_max():
    q = data.load_qrels(TINY)
    assert q[0] == {0: 2, 3: 1, 1: 0}
    assert q[1] == {1: 2, 0: 0}  # labelled Exact and Partial -> Exact
    assert q[2] == {2: 1, 3: 0}  # same label twice -> once


def test_queries_and_split():
    qs = data.load_queries(TINY)
    assert list(qs["query"]) == ["oak coffee table", "green velvet chair", "metal bed"]
    assert qs.loc[2, "query_class"] == ""
    s = data.make_split(range(100), seed=0)
    assert [len(s[k]) for k in ("train", "dev", "test")] == [50, 20, 30]
    assert set(s["train"]) | set(s["dev"]) | set(s["test"]) == set(range(100))
    assert not set(s["train"]) & set(s["test"])
    assert s == data.make_split(range(100), seed=0)


def test_parse_features():
    assert data.parse_features("a : b|c:1| |nokey") == [("a", "b"), ("c", "1")]
    assert data.parse_features(float("nan")) == []
