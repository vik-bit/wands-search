"""BM25 baselines and field-weight tuning (dev queries only).

  bm25_concat   all five fields concatenated into one, k1 = 1.2, b = 0.75 (the untuned baseline)
  bm25_fields   weighted sum of per-field BM25; stemming, k1, b and the five field weights are
                chosen by dev NDCG@10 (coordinate ascent over a weight grid)

Writes results/runs/bm25_*.json (top 100 for all 480 queries), results/bm25_tuning.json and
caches the chosen index in data/cache/bm25_fields.pkl for the later stages.
"""
from __future__ import annotations

import itertools
import json
import pickle
import time

import numpy as np
import pandas as pd

from wands_search import data, metrics
from wands_search.bm25 import FIELDS, BM25Index, tokenise_fields, top_k
from wands_search.runs import CACHE, DEPTH, RAW, ROOT, SPLIT, save_run

GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)


def dev_ndcg(field_scores, weights, qrels, qids, pids) -> float:
    vals = []
    for q in qids:
        s = sum(weights[f] * field_scores[q][f] for f in FIELDS)
        vals.append(metrics.ndcg_at_k(top_k(s, 10, pids), qrels[q], 10))
    return float(np.nanmean(vals))


def coordinate_ascent(field_scores, qrels, qids, pids, passes: int = 3):
    w = {f: 1.0 for f in FIELDS}
    best = dev_ndcg(field_scores, w, qrels, qids, pids)
    for _ in range(passes):
        improved = False
        for f in FIELDS:
            for v in GRID:
                if v == w[f]:
                    continue
                trial = {**w, f: v}
                if all(x == 0 for x in trial.values()):
                    continue
                score = dev_ndcg(field_scores, trial, qrels, qids, pids)
                if score > best + 1e-6:
                    best, w, improved = score, trial, True
        if not improved:
            break
    return w, best


def main():
    products = data.load_products(RAW)
    queries = data.load_queries(RAW)
    qrels = data.load_qrels(RAW)
    split = data.load_split(SPLIT)
    pids = products.product_id.to_numpy()
    qtext = dict(zip(queries.query_id, queries["query"]))
    dev = split["dev"]
    log = {"grid": list(GRID), "configs": []}

    # --- untuned baseline: one concatenated field
    concat = pd.DataFrame({"all": products[list(FIELDS)].agg(" ".join, axis=1)})
    base = BM25Index.build(concat, fields=("all",))
    run = {q: top_k(base.score(t, {"all": 1.0}), DEPTH, pids) for q, t in qtext.items()}
    save_run("bm25_concat", run)
    log["bm25_concat_dev_ndcg10"] = float(np.nanmean(
        [metrics.ndcg_at_k(run[q], qrels[q]) for q in dev]))
    print("concat dev ndcg@10", round(log["bm25_concat_dev_ndcg10"], 4))

    # --- tuned multi-field
    best = None
    for do_stem in (True, False):
        toks = tokenise_fields(products, FIELDS, do_stem)
        for k1, b in itertools.product((0.6, 1.2, 2.0), (0.3, 0.75, 1.0)):
            t0 = time.time()
            idx = BM25Index.from_tokens(toks, k1=k1, b=b)
            fs = {q: idx.field_scores(qtext[q], do_stem) for q in dev}
            w, score = coordinate_ascent(fs, qrels, dev, pids)
            cfg = {"stem": do_stem, "k1": k1, "b": b, "weights": w, "dev_ndcg10": score}
            log["configs"].append(cfg)
            print(cfg, f"{time.time() - t0:.0f}s", flush=True)
            if best is None or score > best[0]["dev_ndcg10"]:
                best = (cfg, idx)
    cfg, idx = best
    log["chosen"] = cfg
    # Same k1/b/stemming with all weights = 1: isolates what the tuned weights add.
    run_eq = {q: top_k(idx.score(t, {f: 1.0 for f in FIELDS}, cfg["stem"]), DEPTH, pids)
              for q, t in qtext.items()}
    save_run("bm25_fields_equal", run_eq)
    run, scores = {}, {}
    for q, t in qtext.items():
        s = idx.score(t, cfg["weights"], cfg["stem"])
        order = top_k(s, DEPTH)
        run[q], scores[q] = list(pids[order]), list(s[order])
    save_run("bm25_fields", run, scores)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "bm25_fields.pkl", "wb") as fh:
        pickle.dump({"index": idx, "cfg": cfg}, fh)
    (ROOT / "results" / "bm25_tuning.json").write_text(json.dumps(log, indent=1))
    print("chosen", cfg)


if __name__ == "__main__":
    main()
