"""Streamlit demo: type a query, see ranked WANDS products with each stage's score.

Pipeline as evaluated: tuned 5-field BM25 + bge-small, weighted min-max fusion with the
dev-chosen BM25 weight, then optionally the cross-encoder on the top N (dev chose N = 10).

    streamlit run app/demo.py

Needs the caches made by scripts/run_bm25.py and scripts/run_dense.py (data/cache/). Uses vecdex
if it is installed, otherwise exact search. If the query is one of the 480 WANDS queries, the
human label of every result is shown too.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wands_search import data, fusion  # noqa: E402
from wands_search.bm25 import top_k  # noqa: E402
from wands_search.dense import Encoder, ExactIndex, product_text  # noqa: E402
from wands_search.runs import CACHE, RAW, ROOT  # noqa: E402

LABEL = {2: "Exact", 1: "Partial", 0: "Irrelevant"}


@st.cache_resource
def load():
    products = data.load_products(RAW)
    with open(CACHE / "bm25_fields.pkl", "rb") as fh:
        bm = pickle.load(fh)
    P = np.load(CACHE / "product_emb.npy")
    try:
        from wands_search.dense import VecdexIndex
        index, backend = VecdexIndex(P), "vecdex HNSW"
    except ImportError:
        index, backend = ExactIndex(P), "exact search"
    queries = data.load_queries(RAW)
    qrels = data.load_qrels(RAW)
    by_text = {t.strip().lower(): q for q, t in zip(queries.query_id, queries["query"])}
    alpha = json.loads((ROOT / "results" / "fusion_tuning.json").read_text())["chosen"]["bm25_weight"]
    return products, bm, Encoder(), index, backend, by_text, qrels, alpha


@st.cache_resource
def reranker():
    from wands_search.rerank import CrossEncoder
    return CrossEncoder()


def main():
    st.set_page_config(page_title="WANDS search demo", layout="wide")
    st.title("WANDS product search")
    st.caption("BM25 (5 fields) + bge-small dense retrieval, weighted fusion, optionally reranked by "
               "a cross-encoder. Data: Wayfair's public WANDS dataset (MIT). A learning project.")
    products, bm, enc, index, backend, by_text, qrels, alpha = load()
    pids = products.product_id.to_numpy()
    q = st.text_input("Query", "modern farmhouse dining table")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Results", 5, 50, 10)
    use_ce = c2.checkbox("Rerank with the cross-encoder", value=False)
    depth = c3.select_slider("Rerank depth", [10, 20, 50], value=10)
    if not q.strip():
        return
    t = {}
    t0 = time.perf_counter()
    s = bm["index"].score(q, bm["cfg"]["weights"], bm["cfg"]["stem"])
    order = top_k(s, 100)
    bm_ids, bm_scores = list(pids[order]), dict(zip(pids[order], s[order]))
    t["BM25"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    qv = enc.queries([q])
    idx, sim = index.search(qv, 100)
    de_ids, de_scores = list(pids[idx[0]]), dict(zip(pids[idx[0]], sim[0]))
    t["dense (encode + " + backend + ")"] = time.perf_counter() - t0
    ids, fused = fusion.weighted([bm_ids, de_ids], [list(s[order]), list(sim[0])], [alpha, 1 - alpha], depth=100)
    fused = dict(zip(ids, fused))
    ce_scores = {}
    if use_ce:
        t0 = time.perf_counter()
        rows = products.set_index("product_id").loc[ids[:depth]]
        sc = reranker().score(q, [product_text(r) for _, r in rows.iterrows()])
        ce_scores = dict(zip(ids[:depth], sc))
        ids = sorted(ids[:depth], key=lambda p: -ce_scores[p]) + ids[depth:]
        t[f"cross-encoder ({depth} pairs)"] = time.perf_counter() - t0
    st.caption(" · ".join(f"{name}: {sec * 1000:.0f} ms" for name, sec in t.items()))

    qid = by_text.get(q.strip().lower())
    if qid is not None:
        st.info(f"This is WANDS query {qid}: results show the human label (unjudged = never labelled).")
    p = products.set_index("product_id")
    rows = []
    for rank, pid in enumerate(ids[:k], start=1):
        r = {"rank": rank, "product": p.at[pid, "name"], "class": p.at[pid, "cls"],
             "bm25": round(float(bm_scores.get(pid, 0)), 2), "dense cos": round(float(de_scores.get(pid, np.nan)), 3),
             "fused": round(fused[pid], 3)}
        if use_ce:
            r["cross-enc"] = round(ce_scores.get(pid, np.nan), 2)
        if qid is not None:
            g = qrels.get(qid, {}).get(int(pid))
            r["label"] = LABEL.get(g, "unjudged")
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


main()
