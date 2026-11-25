"""Saving, loading and evaluating runs (query_id -> ranked product ids, plus optional scores)."""
from __future__ import annotations

import json
from pathlib import Path

from . import metrics, stats

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
RUNS = ROOT / "results" / "runs"
SPLIT = ROOT / "splits" / "queries_seed0.json"
DEPTH = 100  # every system returns its top 100 for every query


def save_run(name: str, run: dict, scores: dict | None = None) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    payload = {"ids": {str(q): [int(p) for p in r] for q, r in run.items()}}
    if scores is not None:
        payload["scores"] = {str(q): [round(float(s), 5) for s in v] for q, v in scores.items()}
    path = RUNS / f"{name}.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def load_run(name: str, with_scores: bool = False):
    payload = json.loads((RUNS / f"{name}.json").read_text())
    run = {int(q): r for q, r in payload["ids"].items()}
    if with_scores:
        return run, {int(q): s for q, s in payload.get("scores", {}).items()}
    return run


def table(systems: dict, qrels: dict, query_ids, baseline: str | None = None,
          metric_names=("ndcg@10", "map", "recall@50", "mrr", "judged@10")) -> dict:
    """Mean, 95% bootstrap CI and (if baseline is given) a paired comparison for every system."""
    out = {}
    per_q = {name: metrics.evaluate(run, qrels, query_ids) for name, run in systems.items()}
    for name, pq in per_q.items():
        row = {}
        for m in metric_names:
            mean, lo, hi = stats.bootstrap_ci(pq[m].to_numpy())
            row[m] = {"mean": mean, "ci": [lo, hi], "n": int(pq[m].notna().sum())}
        if baseline and name != baseline:
            row["vs_" + baseline] = stats.compare(per_q[baseline]["ndcg@10"].to_numpy(),
                                                   pq["ndcg@10"].to_numpy())
        out[name] = row
    return out


def fmt_table(tab: dict, metric_names=("ndcg@10", "map", "recall@50", "mrr")) -> str:
    lines = ["| system | " + " | ".join(metric_names) + " | Δ NDCG@10 vs baseline (95% CI), p |",
             "|---" * (len(metric_names) + 2) + "|"]
    for name, row in tab.items():
        cells = [f"{row[m]['mean']:.3f} ({row[m]['ci'][0]:.3f}–{row[m]['ci'][1]:.3f})" for m in metric_names]
        cmp_key = next((k for k in row if k.startswith("vs_")), None)
        if cmp_key:
            c = row[cmp_key]
            cells.append(f"{c['delta']:+.3f} ({c['ci_low']:+.3f} to {c['ci_high']:+.3f}), p={c['p_value']:.4f}")
        else:
            cells.append("—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
