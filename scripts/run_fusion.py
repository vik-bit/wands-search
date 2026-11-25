"""Hybrid BM25 + dense: RRF (k tuned on dev) and weighted min-max fusion (weight tuned on dev)."""
from __future__ import annotations

import json

import numpy as np

from wands_search import data, fusion, metrics
from wands_search.runs import DEPTH, RAW, ROOT, SPLIT, load_run, save_run


def dev_score(run, qrels, dev):
    return float(np.nanmean([metrics.ndcg_at_k(run[q], qrels[q]) for q in dev]))


def main():
    qrels = data.load_qrels(RAW)
    dev = data.load_split(SPLIT)["dev"]
    bm, bm_s = load_run("bm25_fields", with_scores=True)
    de, de_s = load_run("dense_vecdex", with_scores=True)
    qids = sorted(bm)
    log = {"rrf": {}, "weighted": {}}

    def rrf_run(k):
        out = {q: fusion.rrf([bm[q], de[q]], k=k, depth=DEPTH) for q in qids}
        return {q: v[0] for q, v in out.items()}, {q: v[1] for q, v in out.items()}

    for k in (1, 10, 30, 60, 100):
        log["rrf"][k] = dev_score(rrf_run(k)[0], qrels, dev)
    best_k = max(log["rrf"], key=log["rrf"].get)
    save_run("hybrid_rrf", *rrf_run(best_k))

    def w_run(alpha):
        out = {q: fusion.weighted([bm[q], de[q]], [bm_s[q], de_s[q]], [alpha, 1 - alpha], depth=DEPTH)
               for q in qids}
        return {q: v[0] for q, v in out.items()}, {q: v[1] for q, v in out.items()}

    for a in np.round(np.arange(0, 1.01, 0.1), 2):
        log["weighted"][float(a)] = dev_score(w_run(a)[0], qrels, dev)
    best_a = max(log["weighted"], key=log["weighted"].get)
    save_run("hybrid_weighted", *w_run(best_a))
    log["chosen"] = {"rrf_k": best_k, "bm25_weight": best_a}
    print(json.dumps(log, indent=1))
    (ROOT / "results" / "fusion_tuning.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
