"""Leave-one-feature-group-out ablation of LambdaMART on test, with paired CIs vs the full model."""
from __future__ import annotations

import json

import numpy as np

from wands_search import data, metrics, stats
from wands_search.runs import RAW, ROOT, SPLIT, load_run

GROUPS = ["bm25", "dense", "rank", "coverage", "class", "reviews", "query"]


def main():
    qrels = data.load_qrels(RAW)
    test = data.load_split(SPLIT)["test"]
    full = metrics.evaluate(load_run("lambdamart"), qrels, test)["ndcg@10"].to_numpy()
    out = {}
    lines = ["| removed group | test NDCG@10 | Δ vs full model (95% CI) | p |", "|---|---|---|---|"]
    for g in GROUPS:
        x = metrics.evaluate(load_run(f"ablate_minus_{g}"), qrels, test)["ndcg@10"].to_numpy()
        c = stats.compare(full, x)
        out[g] = {"ndcg@10": float(np.nanmean(x)), **c}
        lines.append(f"| {g} | {np.nanmean(x):.3f} | {c['delta']:+.3f} ({c['ci_low']:+.3f} to {c['ci_high']:+.3f}) | {c['p_value']:.3f} |")
    (ROOT / "results" / "ablation.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
