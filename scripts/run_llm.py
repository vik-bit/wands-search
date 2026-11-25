"""Where LLM query understanding helps and where it hurts.

    python scripts/run_llm.py --backend groq   --model llama-3.3-70b-versatile
    python scripts/run_llm.py --backend ollama --model qwen2.5:3b
    python scripts/run_llm.py --backend cache  --model qwen2.5-7b-instruct-4bit   (Kaggle output)

The LLM output for every query is cached in results/llm/<model>.jsonl. Each variant below
changes only the query side and reuses the same first stage (tuned BM25 + bge-small via vecdex,
weighted fusion with the dev-chosen weight):

  original      the query as typed (reproduces hybrid_weighted)
  corrected     the LLM's spelling-corrected query
  rewrite       the LLM's catalogue-style rewrite, used instead of the query
  orig+rewrite  RRF of the original and the rewrite result lists (expansion, not replacement)
  attr_boost    original list, re-scored: + beta * share of extracted colour/material words found in
                the product's name + attributes (beta chosen on dev)
  attr_filter   original list with products that lack an extracted colour/material word removed
"""
from __future__ import annotations

import argparse
import json
import pickle
import re

import numpy as np

from wands_search import data, fusion, metrics, stats
from wands_search.bm25 import top_k
from wands_search.dense import Encoder, VecdexIndex
from wands_search.llm import parse, run_all
from wands_search.runs import CACHE, DEPTH, RAW, ROOT, SPLIT, save_run
from wands_search.text import tokenize

BETAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=("groq", "ollama", "cache"))
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    cache_path = ROOT / "results" / "llm" / f"{tag}.jsonl"

    products = data.load_products(RAW)
    queries = data.load_queries(RAW)
    qrels = data.load_qrels(RAW)
    split = data.load_split(SPLIT)
    qtext = dict(zip(queries.query_id, queries["query"]))
    if args.backend == "cache":
        recs = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
        llm = {r["query_id"]: {**parse(r["raw"], r["query"]), "seconds": r["seconds"]} for r in recs}
        missing = set(qtext) - set(llm)
        if missing:
            raise SystemExit(f"{len(missing)} queries missing from {cache_path}")
    else:
        llm = run_all(qtext, args.backend, args.model, cache_path, sleep=2.0 if args.backend == "groq" else 0)

    pids = products.product_id.to_numpy()
    with open(CACHE / "bm25_fields.pkl", "rb") as fh:
        bm = pickle.load(fh)
    alpha = json.loads((ROOT / "results" / "fusion_tuning.json").read_text())["chosen"]["bm25_weight"]
    P = np.load(CACHE / "product_emb.npy")
    enc, vx = Encoder(), VecdexIndex(P, ef=200)

    def hybrid(texts: dict):
        qids = list(texts)
        Q = enc.queries([texts[q] for q in qids])
        idx, sim = vx.search(Q, DEPTH)
        out = {}
        for q, ii, ss in zip(qids, idx, sim):
            s = bm["index"].score(texts[q], bm["cfg"]["weights"], bm["cfg"]["stem"])
            o = top_k(s, DEPTH)
            ids, sc = fusion.weighted([list(pids[o]), list(pids[ii])], [list(s[o]), list(ss)],
                                      [alpha, 1 - alpha], depth=DEPTH)
            out[q] = (ids, sc)
        return out

    orig = hybrid(qtext)
    corrected = hybrid({q: llm[q]["corrected"] for q in qtext})
    rewrite = hybrid({q: llm[q]["rewrite"] for q in qtext})
    runs = {"original": {q: v[0] for q, v in orig.items()},
            "corrected": {q: v[0] for q, v in corrected.items()},
            "rewrite": {q: v[0] for q, v in rewrite.items()},
            "orig+rewrite": {q: fusion.rrf([orig[q][0], rewrite[q][0]], k=60, depth=DEPTH)[0] for q in qtext}}

    # attribute words the LLM extracted (colour + material), as tokens
    attr_words = {q: set(tokenize(" ".join(x for x in (llm[q]["color"], llm[q]["material"]) if x)))
                  for q in qtext}
    p = products.set_index("product_id")
    ptoks = {}

    def toks(pid):
        if pid not in ptoks:
            ptoks[pid] = set(tokenize(p.at[pid, "name"] + " " + p.at[pid, "attrs"]))
        return ptoks[pid]

    def boosted(beta):
        out = {}
        for q, (ids, sc) in orig.items():
            w = attr_words[q]
            if not w or beta == 0:
                out[q] = ids
                continue
            new = [s + beta * len(w & toks(d)) / len(w) for d, s in zip(ids, sc)]
            out[q] = [d for d, _ in sorted(zip(ids, new), key=lambda x: -x[1])]
        return out

    dev_beta = {b: float(np.nanmean([metrics.ndcg_at_k(boosted(b)[q], qrels[q]) for q in split["dev"]]))
                for b in BETAS}
    beta = max(dev_beta, key=dev_beta.get)
    runs["attr_boost"] = boosted(beta)
    runs["attr_filter"] = {q: ([d for d in ids if attr_words[q] <= toks(d)] if attr_words[q] else ids)
                           for q, (ids, _) in orig.items()}

    test = split["test"]
    per_q = {k: metrics.evaluate(r, qrels, test) for k, r in runs.items()}
    changed = {"corrected": [q for q in qtext if llm[q]["corrected"].strip().lower() != qtext[q].strip().lower()],
               "rewrite": [q for q in qtext if llm[q]["rewrite"].strip().lower() != qtext[q].strip().lower()],
               "attr": [q for q in qtext if attr_words[q]]}
    res = {"model": args.model, "backend": args.backend, "attr_boost_beta": beta, "dev_beta": dev_beta,
           "parse_ok": sum(llm[q]["parse_ok"] for q in qtext), "n_queries": len(qtext),
           "seconds_per_query": {"p50": float(np.median([llm[q]["seconds"] for q in qtext])),
                                 "p95": float(np.quantile([llm[q]["seconds"] for q in qtext], 0.95))},
           "n_changed_all": {k: len(v) for k, v in changed.items()},
           "n_changed_test": {k: len(set(v) & set(test)) for k, v in changed.items()},
           "test": {}}
    for name, pq in per_q.items():
        row = {m: float(pq[m].mean()) for m in ("ndcg@10", "recall@50", "mrr")}
        if name != "original":
            row["vs_original"] = stats.compare(per_q["original"]["ndcg@10"].to_numpy(), pq["ndcg@10"].to_numpy())
        res["test"][name] = row
    # examples: biggest wins and losses on the test queries for each replacement variant
    ex = {}
    for name in ("corrected", "rewrite", "orig+rewrite", "attr_filter"):
        d = (per_q[name]["ndcg@10"] - per_q["original"]["ndcg@10"]).dropna().sort_values()
        pick = list(d.index[:4]) + list(d.index[-4:][::-1])
        ex[name] = [{"query": qtext[q], "llm": {k: llm[q][k] for k in ("corrected", "rewrite", "color", "material",
                                                                        "size", "style", "product_type")},
                     "ndcg_original": float(per_q["original"].at[q, "ndcg@10"]),
                     "ndcg_variant": float(per_q[name].at[q, "ndcg@10"])} for q in pick if abs(d[q]) > 1e-9]
    res["examples"] = ex
    for name, r in runs.items():
        save_run(f"llm_{tag}_{name}", r)
    out = ROOT / "results" / f"llm_{tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
