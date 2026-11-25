"""Error analysis on the test queries: NDCG@10 by query type, and the worst queries with what
the best system put at the top. Writes results/error_analysis.json and results/error_analysis.md.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from wands_search import data, metrics
from wands_search.query_tags import tags
from wands_search.runs import RAW, ROOT, SPLIT, load_run
from wands_search.text import tokenize

SYSTEMS = ["bm25_fields", "dense_vecdex", "hybrid_rrf", "rerank_ms-marco-MiniLM-L-6-v2", "lambdamart"]
LABEL = {2: "E", 1: "P", 0: "I", None: "?"}


def main():
    products = data.load_products(RAW).set_index("product_id")
    queries = data.load_queries(RAW).set_index("query_id")
    qrels = data.load_qrels(RAW)
    test = data.load_split(SPLIT)["test"]
    df = Counter()
    for col in ("name", "cls", "hierarchy", "attrs"):
        for t in products[col]:
            df.update(set(tokenize(t, do_stem=False)))
    vocab = {t for t, c in df.items() if c >= 3}
    qtags = {q: tags(queries.at[q, "query"], vocab) for q in test}
    per_q = {s: metrics.evaluate(load_run(s), qrels, test)["ndcg@10"] for s in SYSTEMS}

    groups = {}
    for tag in next(iter(qtags.values())):
        qs = [q for q in test if qtags[q][tag]]
        if not qs:
            continue
        groups[tag] = {"n": len(qs), **{s: float(np.nanmean(per_q[s].loc[qs])) for s in SYSTEMS}}
    out = {"groups": groups, "tags": {int(q): t for q, t in qtags.items()}}

    best = "lambdamart"
    run = load_run(best)
    worst = per_q[best].sort_values().index[:12]
    examples = []
    for q in worst:
        j = qrels[q]
        top = run[q][:5]
        examples.append({"query_id": int(q), "query": queries.at[q, "query"],
                         "query_class": queries.at[q, "query_class"],
                         "ndcg10": float(per_q[best].loc[q]),
                         "n_exact": sum(g == 2 for g in j.values()),
                         "n_partial": sum(g == 1 for g in j.values()),
                         "top5": [f"[{LABEL[j.get(p)]}] {products.at[p, 'name']} ({products.at[p, 'cls']})"
                                  for p in top]})
    out["worst_" + best] = examples
    (ROOT / "results" / "error_analysis.json").write_text(json.dumps(out, indent=1))

    md = ["NDCG@10 on test queries by query type (rule-based tags, a query can have several).", "",
          "| query type | n | " + " | ".join(SYSTEMS) + " |", "|---" * (len(SYSTEMS) + 2) + "|"]
    for tag, g in groups.items():
        md.append(f"| {tag} | {g['n']} | " + " | ".join(f"{g[s]:.3f}" for s in SYSTEMS) + " |")
    md += ["", f"Worst test queries for {best} (E = Exact, P = Partial, I = Irrelevant, ? = unjudged):", ""]
    for e in examples:
        md.append(f"- **{e['query']}** (class {e['query_class']!r}; {e['n_exact']} Exact, "
                  f"{e['n_partial']} Partial judged) NDCG@10 = {e['ndcg10']:.3f}")
        md += [f"    - {t}" for t in e["top5"]]
    (ROOT / "results" / "error_analysis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
