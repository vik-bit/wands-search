"""Final table on the 144 test queries: every system, 95% bootstrap CIs, paired tests.

Each system is compared (NDCG@10, paired bootstrap CI of the difference + randomization test)
with the untuned BM25 baseline, and each step of the pipeline with the step before it.
Writes results/test_results.json and results/test_results.md.
"""
from __future__ import annotations

import json

from wands_search import data, metrics, stats
from wands_search.runs import RAW, ROOT, RUNS, SPLIT, fmt_table, load_run, table

SYSTEMS = ["bm25_concat", "bm25_fields_equal", "bm25_fields", "dense_exact", "dense_vecdex",
           "hybrid_rrf", "hybrid_weighted", "rerank_ms-marco-MiniLM-L-6-v2",
           "rerank_bge-reranker-base", "pointwise", "lambdamart", "lambdamart_ce",
           "lambdamart_oracle_class"]
# rerank and LTR candidates are the hybrid_weighted top 100 (it won on dev), so they are compared to it
STEPS = [("bm25_concat", "bm25_fields"), ("bm25_fields_equal", "bm25_fields"),
         ("bm25_fields", "hybrid_weighted"), ("dense_vecdex", "hybrid_weighted"),
         ("hybrid_rrf", "hybrid_weighted"), ("hybrid_weighted", "rerank_ms-marco-MiniLM-L-6-v2"),
         ("hybrid_weighted", "rerank_bge-reranker-base"), ("hybrid_weighted", "lambdamart"), ("pointwise", "lambdamart"),
         ("lambdamart", "lambdamart_ce"), ("rerank_ms-marco-MiniLM-L-6-v2", "lambdamart_ce"),
         ("dense_exact", "dense_vecdex"), ("lambdamart", "lambdamart_oracle_class")]


def main():
    qrels = data.load_qrels(RAW)
    test = data.load_split(SPLIT)["test"]
    systems = {s: load_run(s) for s in SYSTEMS if (RUNS / f"{s}.json").exists()}
    tab = table(systems, qrels, test, baseline="bm25_concat")
    steps = {}
    per_q = {s: metrics.evaluate(r, qrels, test) for s, r in systems.items()}
    for a, b in STEPS:
        if a in per_q and b in per_q:
            steps[f"{a} -> {b}"] = {m: stats.compare(per_q[a][m].to_numpy(), per_q[b][m].to_numpy())
                                    for m in ("ndcg@10", "recall@50")}
    (ROOT / "results" / "test_results.json").write_text(json.dumps({"table": tab, "steps": steps}, indent=1))
    md = [f"Test split: {len(test)} queries. Mean (95% bootstrap CI over queries).", "",
          fmt_table(tab), "", "| step | Δ NDCG@10 (95% CI) | p | wins/losses/ties | Δ recall@50 |",
          "|---|---|---|---|---|"]
    for k, v in steps.items():
        n, r = v["ndcg@10"], v["recall@50"]
        md.append(f"| {k} | {n['delta']:+.3f} ({n['ci_low']:+.3f} to {n['ci_high']:+.3f}) | {n['p_value']:.4f} "
                  f"| {n['wins']}/{n['losses']}/{n['ties']} | {r['delta']:+.3f} |")
    md += ["", "| system | judged@10 |", "|---|---|"]
    md += [f"| {s} | {row['judged@10']['mean']:.3f} |" for s, row in tab.items()]
    (ROOT / "results" / "test_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
