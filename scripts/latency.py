"""Per-query latency of each stage, measured one query at a time on the 144 test queries.

Laptop CPU (Intel i7-1255U under WSL2), no batching across queries, everything warm (models
loaded, one warm-up query first). Reports p50 / p95 / mean in milliseconds.
"""
from __future__ import annotations

import json
import pickle
import time

import lightgbm as lgb  # noqa: F401  (loaded so its import time is not measured)
import numpy as np

from wands_search import data, fusion
from wands_search.bm25 import top_k
from wands_search.dense import Encoder, ExactIndex, VecdexIndex, product_text
from wands_search.ltr import build_features, columns, token_sets
from wands_search.rerank import CrossEncoder, rerank
from wands_search.runs import CACHE, RAW, ROOT, SPLIT


def ms(xs):
    a = np.array(xs) * 1000
    return {"p50": round(float(np.median(a)), 2), "p95": round(float(np.quantile(a, 0.95)), 2),
            "mean": round(float(a.mean()), 2)}


def main():
    products = data.load_products(RAW)
    queries = data.load_queries(RAW).set_index("query_id")
    test = data.load_split(SPLIT)["test"]
    pids = products.product_id.to_numpy()
    with open(CACHE / "bm25_fields.pkl", "rb") as fh:
        bm = pickle.load(fh)
    P = np.load(CACHE / "product_emb.npy")
    enc, ce = Encoder(), CrossEncoder()
    exact, vx = ExactIndex(P), VecdexIndex(P, ef=200)
    fus = json.loads((ROOT / "results" / "fusion_tuning.json").read_text())["chosen"]
    ltr = pickle.load(open(CACHE / "lambdamart_model.pkl", "rb"))
    text = {p: product_text(r) for p, r in products.set_index("product_id").iterrows()}
    pid_to_row = {p: i for i, p in enumerate(pids)}
    name, allf, cls = token_sets(products)
    T = {k: [] for k in ("bm25", "encode_query", "vecdex_search", "exact_search", "rrf",
                         "cross_encoder_top50", "ltr_features_and_predict", "total_lambdamart_pipeline")}
    for i, q in enumerate([test[0]] + test):
        text_q = queries.at[q, "query"]
        t0 = time.perf_counter()
        s = bm["index"].score(text_q, bm["cfg"]["weights"], bm["cfg"]["stem"])
        o = top_k(s, 100)
        bm_ids = list(pids[o])
        t1 = time.perf_counter()
        qv = enc.queries([text_q])
        t2 = time.perf_counter()
        idx, _ = vx.search(qv, 100)
        t3 = time.perf_counter()
        exact.search(qv, 100)
        t4 = time.perf_counter()
        ids, _ = fusion.rrf([bm_ids, list(pids[idx[0]])], k=fus["rrf_k"], depth=100)
        t5 = time.perf_counter()
        sc = ce.score(text_q, [text[p] for p in ids[:50]])
        rerank(ids, sc, 50)
        t6 = time.perf_counter()
        f = build_features(q, text_q, "", ids, bm25_index=bm["index"], bm25_cfg=bm["cfg"],
                           pid_to_row=pid_to_row, products=products, query_vec=qv[0], product_vecs=P,
                           name_tokens=name, all_tokens=allf, cls_tokens=cls)
        ltr["model"].predict(f[ltr["columns"]], num_iteration=ltr["trees"])
        t7 = time.perf_counter()
        if i == 0:
            continue  # warm-up
        T["bm25"].append(t1 - t0)
        T["encode_query"].append(t2 - t1)
        T["vecdex_search"].append(t3 - t2)
        T["exact_search"].append(t4 - t3)
        T["rrf"].append(t5 - t4)
        T["cross_encoder_top50"].append(t6 - t5)
        T["ltr_features_and_predict"].append(t7 - t6)
        T["total_lambdamart_pipeline"].append((t3 - t0) + (t5 - t4) + (t7 - t6))
    out = {k: ms(v) for k, v in T.items()}
    out["n_queries"] = len(test)
    (ROOT / "results" / "latency.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
