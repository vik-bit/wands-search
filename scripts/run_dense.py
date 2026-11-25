"""Embed all products once, then dense retrieval with exact search and with vecdex.

Writes results/runs/dense_exact.json, results/runs/dense_vecdex.json, results/dense_check.json
(vecdex vs exact: top-k overlap and NDCG difference at several ef), and caches the embeddings in
data/cache/ for the later stages.
"""
from __future__ import annotations

import json
import time

import numpy as np

from wands_search import data, metrics
from wands_search.dense import MODEL, Encoder, ExactIndex, VecdexIndex, overlap_at_k, product_text
from wands_search.runs import CACHE, DEPTH, RAW, ROOT, SPLIT, save_run


def main():
    products = data.load_products(RAW)
    queries = data.load_queries(RAW)
    qrels = data.load_qrels(RAW)
    split = data.load_split(SPLIT)
    pids = products.product_id.to_numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    enc = Encoder()

    pfile = CACHE / "product_emb.npy"
    if pfile.exists():
        P = np.load(pfile)
    else:
        t0 = time.time()
        texts = [product_text(r) for _, r in products.iterrows()]
        P = enc.products(texts)
        np.save(pfile, P)
        print(f"embedded {len(P)} products in {time.time() - t0:.0f}s")
    t0 = time.time()
    Q = enc.queries(queries["query"].tolist())
    print(f"embedded {len(Q)} queries in {time.time() - t0:.1f}s")
    np.save(CACHE / "query_emb.npy", Q)
    qids = queries.query_id.tolist()

    exact_idx, exact_sim = ExactIndex(P).search(Q, DEPTH)
    run_exact = {q: list(pids[i]) for q, i in zip(qids, exact_idx)}
    save_run("dense_exact", run_exact, {q: list(s) for q, s in zip(qids, exact_sim)})

    t0 = time.time()
    vx = VecdexIndex(P, M=16, ef_construction=200)
    build_s = time.time() - t0
    check = {"model": MODEL, "dim": int(P.shape[1]), "n_products": int(len(P)),
             "M": 16, "ef_construction": 200, "build_seconds": round(build_s, 1), "by_ef": []}
    test = split["test"]
    ndcg_exact = np.nanmean([metrics.ndcg_at_k(run_exact[q], qrels[q]) for q in test])
    for ef in (100, 200, 400):
        idx, sim = vx.search(Q, DEPTH, ef=ef)
        run = {q: list(pids[i]) for q, i in zip(qids, idx)}
        ndcg = np.nanmean([metrics.ndcg_at_k(run[q], qrels[q]) for q in test])
        row = {"ef": ef, "overlap@10": overlap_at_k(idx, exact_idx, 10),
               "overlap@100": overlap_at_k(idx, exact_idx, DEPTH),
               "test_ndcg10_vecdex": float(ndcg), "test_ndcg10_exact": float(ndcg_exact),
               "queries_top10_identical": int(sum(list(a[:10]) == list(b[:10])
                                                   for a, b in zip(idx, exact_idx)))}
        check["by_ef"].append(row)
        print(row)
        if ef == 200:
            save_run("dense_vecdex", run, {q: list(s) for q, s in zip(qids, sim)})
    (ROOT / "results" / "dense_check.json").write_text(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
