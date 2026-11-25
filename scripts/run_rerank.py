"""Cross-encoder reranking of the hybrid top 100.

Scores every (query, candidate) pair in the chosen hybrid run's top 100 once and caches them
(data/cache/ce_<model>.json; LambdaMART reuses them as a feature). The rerank depth is chosen on
dev. Usage:  python scripts/run_rerank.py [model] [--dev-test] [--max-depth=N]
  --dev-test     score only dev and test queries (for a slower model)
  --max-depth=N  score only the top N candidates (depths above N are not tried)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from wands_search import data, metrics
from wands_search.dense import product_text
from wands_search.rerank import MODEL, CrossEncoder, rerank
from wands_search.runs import CACHE, RAW, ROOT, SPLIT, load_run, save_run

DEPTHS = (10, 20, 50, 100)


def main():
    model = next((a for a in sys.argv[1:] if not a.startswith("--")), MODEL)
    dev_test = "--dev-test" in sys.argv
    max_depth = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--max-depth=")), 100)
    tag = model.split("/")[-1]
    products = data.load_products(RAW).set_index("product_id")
    queries = data.load_queries(RAW)
    qrels = data.load_qrels(RAW)
    split = data.load_split(SPLIT)
    qtext = dict(zip(queries.query_id, queries["query"]))
    tuning = json.loads((ROOT / "results" / "fusion_tuning.json").read_text())
    base_name = "hybrid_rrf" if max(tuning["rrf"].values()) >= max(tuning["weighted"].values()) \
        else "hybrid_weighted"
    base = load_run(base_name)
    text = {pid: product_text(r) for pid, r in products.iterrows()}

    cache_file = CACHE / f"ce_{tag}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    ce = CrossEncoder(model)
    todo = sorted(split["dev"] + split["test"]) if dev_test else sorted(base)
    seconds = {}
    for i, q in enumerate(todo):
        if str(q) in cache:
            continue
        t0 = time.perf_counter()
        cache[str(q)] = ce.score(qtext[q], [text[p] for p in base[q][:max_depth]])
        seconds[q] = time.perf_counter() - t0
        if i % 50 == 0:
            print(i, f"{seconds[q]:.2f}s", flush=True)
            cache_file.write_text(json.dumps(cache))
    cache_file.write_text(json.dumps(cache))
    scores = {int(q): s for q, s in cache.items()}

    log = {"model": model, "base": base_name, "max_depth_scored": max_depth, "dev": {}}
    for d in [d for d in DEPTHS if d <= max_depth]:
        run = {q: rerank(base[q], scores[q], d) for q in split["dev"]}
        log["dev"][d] = float(np.nanmean([metrics.ndcg_at_k(run[q], qrels[q]) for q in split["dev"]]))
    depth = max(log["dev"], key=log["dev"].get)
    log["chosen_depth"] = depth
    if seconds:
        s = np.array(list(seconds.values()))
        log[f"seconds_per_query_top{max_depth}"] = {"p50": float(np.median(s)), "p95": float(np.quantile(s, 0.95))}
    run = {q: rerank(base[q], scores[q], depth) for q in scores}
    save_run(f"rerank_{tag}", run)
    out = ROOT / "results" / ("rerank_MiniLM.json" if tag == MODEL.split("/")[-1] else f"rerank_{tag}.json")
    out.write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
