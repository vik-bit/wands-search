import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wands_search import data
from wands_search.bm25 import BM25Index, top_k
from wands_search.text import stem, tokenize

TINY = Path(__file__).parent / "fixtures" / "tiny"


def test_stemmer():
    assert [stem(w) for w in ["lamps", "benches", "shelves", "bodies", "glass", "mattress", "boxes"]] == \
        ["lamp", "bench", "shelf", "body", "glass", "mattress", "box"]
    assert tokenize("Bar-Stools, 24\"!") == ["bar", "stool", "24"]
    assert tokenize("Bar-Stools", do_stem=False) == ["bar", "stools"]


def test_bm25_weight_by_hand():
    # Two docs, one field. "oak" appears once in doc 0 (length 2), never in doc 1 (length 1).
    docs = pd.DataFrame({"name": ["oak table", "chair"]})
    idx = BM25Index.build(docs, fields=("name",), k1=1.2, b=0.75)
    s = idx.score("oak", {"name": 1.0})
    # idf = log(1 + (2 - 1 + 0.5) / (1 + 0.5)) = log 2 ; avglen = 1.5
    # tf part = 1 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / 1.5)) = 2.2 / 2.5 = 0.88
    assert s[0] == pytest.approx(math.log(2) * 0.88)
    assert s[1] == 0.0


def test_field_weights_change_ranking():
    p = data.load_products(TINY)
    idx = BM25Index.build(p)
    # "oak": name/attrs of product 0 vs description/attrs of product 3
    by_desc = idx.score("oak", {"description": 1.0})
    assert top_k(by_desc, 1)[0] == 3
    by_name = idx.score("oak coffee", {"name": 1.0})
    assert top_k(by_name, 1)[0] == 0
    # unknown words score zero everywhere and return no results
    assert len(top_k(idx.score("zzz", {"name": 1.0}), 3)) == 0


def test_top_k_order_and_ids():
    s = np.array([0.1, 0.0, 0.7, 0.3])
    assert list(top_k(s, 3)) == [2, 3, 0]
    assert top_k(s, 10, ids=["a", "b", "c", "d"]) == ["c", "d", "a"]
