"""LambdaMART (LightGBM lambdarank) on the hybrid top 100, trained on train queries only.

Choices made on dev: how to label unjudged candidates (0 or dropped), tree size, number of
trees (early stopping). The test queries are only used for the final numbers and the ablations.
Also trains a pointwise regressor on the same features with the same search, to compare.
"""
from __future__ import annotations

import itertools
import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from wands_search import data, metrics
from wands_search.ltr import GROUPS, build_features, columns, token_sets
from wands_search.runs import CACHE, RAW, ROOT, SPLIT, load_run, save_run

BASE_GROUPS = ["bm25", "dense", "rank", "coverage", "class", "reviews", "query"]


def features(products, queries, base):
    path = CACHE / "ltr_features.pkl"
    if path.exists():
        return pd.read_pickle(path)
    with open(CACHE / "bm25_fields.pkl", "rb") as fh:
        b = pickle.load(fh)
    P, Q = np.load(CACHE / "product_emb.npy"), np.load(CACHE / "query_emb.npy")
    qrow = {q: i for i, q in enumerate(queries.query_id)}
    ce = {int(q): s for q, s in json.loads((CACHE / "ce_ms-marco-MiniLM-L-6-v2.json").read_text()).items()}
    pid_to_row = {p: i for i, p in enumerate(products.product_id)}
    name, allf, cls = token_sets(products)
    frames = []
    for q, text, qc in queries[["query_id", "query", "query_class"]].itertuples(index=False):
        frames.append(build_features(q, text, qc, base[q], bm25_index=b["index"], bm25_cfg=b["cfg"],
                                     pid_to_row=pid_to_row, products=products, query_vec=Q[qrow[q]],
                                     product_vecs=P, name_tokens=name, all_tokens=allf,
                                     cls_tokens=cls, ce_scores=ce.get(q)))
    df = pd.concat(frames, ignore_index=True)
    df.to_pickle(path)
    return df


def ranked(df, scores):
    out = {}
    tmp = df[["query_id", "product_id"]].assign(s=scores)
    for q, g in tmp.groupby("query_id", sort=False):
        out[int(q)] = g.sort_values("s", ascending=False, kind="stable").product_id.tolist()
    return out


def ndcg(run, qrels, qids):
    return float(np.nanmean([metrics.ndcg_at_k(run[q], qrels[q]) for q in qids]))


def fit(df, cols, train_q, dev_q, qrels, params, objective, unjudged, rounds=1500):
    tr = df[df.query_id.isin(train_q)]
    if unjudged == "drop":
        tr = tr[tr.label.notna()]
    tr = tr.sort_values("query_id", kind="stable")
    y = tr.label.fillna(0).astype(int)
    dv = df[df.query_id.isin(dev_q)]
    common = dict(n_estimators=rounds, learning_rate=0.05, random_state=0, deterministic=True,
                  verbose=-1, n_jobs=8, **params)
    if objective == "lambdarank":
        m = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1, 2], **common)
        m.fit(tr[cols], y, group=tr.groupby("query_id", sort=False).size().to_numpy())
    else:
        m = lgb.LGBMRegressor(objective="regression", **common)
        m.fit(tr[cols], y)
    # choose the number of trees by dev NDCG@10 (same procedure for both objectives)
    best = (-1.0, 0)
    for n in list(range(25, rounds + 1, 25)):
        s = ndcg(ranked(dv, m.predict(dv[cols], num_iteration=n)), qrels, dev_q)
        if s > best[0] + 1e-9:
            best = (s, n)
    return m, best[1], best[0]


def search(df, cols, split, qrels, objective, unjudged_options=("zero", "drop")):
    best = None
    log = []
    for unj, leaves, mcs in itertools.product(unjudged_options, (7, 15, 31), (20, 100)):
        params = {"num_leaves": leaves, "min_child_samples": mcs}
        m, n, s = fit(df, cols, split["train"], split["dev"], qrels, params, objective, unj)
        log.append({"unjudged": unj, **params, "trees": n, "dev_ndcg10": s})
        print(objective, log[-1], flush=True)
        if best is None or s > best[2]:
            best = (m, n, s, {"unjudged": unj, **params})
    return best, log


def main():
    products = data.load_products(RAW)
    queries = data.load_queries(RAW)
    qrels = data.load_qrels(RAW)
    split = data.load_split(SPLIT)
    base_name = json.loads((ROOT / "results" / "rerank_MiniLM.json").read_text())["base"]
    df = features(products, queries, load_run(base_name))
    df["label"] = [qrels[q].get(p, np.nan) for q, p in zip(df.query_id, df.product_id)]
    test_df = df[df.query_id.isin(split["test"])]
    out = {"base_run": base_name, "variants": {}}

    def finish(name, model, n, cols):
        scores = model.predict(df[cols], num_iteration=n)
        save_run(name, ranked(df, scores))
        return float(ndcg(ranked(test_df, model.predict(test_df[cols], num_iteration=n)), qrels, split["test"]))

    cols = columns(BASE_GROUPS)
    (m, n, s, cfg), log = search(df, cols, split, qrels, "lambdarank")
    out["variants"]["lambdamart"] = {"groups": BASE_GROUPS, "config": cfg, "trees": n, "dev_ndcg10": s,
                                     "search": log, "test_ndcg10": finish("lambdamart", m, n, cols)}
    with open(CACHE / "lambdamart_model.pkl", "wb") as fh:
        pickle.dump({"model": m, "trees": n, "columns": cols}, fh)
    imp = dict(zip(cols, m.booster_.feature_importance("gain", iteration=n)))
    out["importance_gain"] = {k: float(v / sum(imp.values())) for k, v in sorted(imp.items(), key=lambda x: -x[1])}
    unj = cfg["unjudged"]

    # pointwise regression with the same features and the same search
    (mp, npw, sp, cfgp), logp = search(df, cols, split, qrels, "regression", (unj,))
    out["variants"]["pointwise"] = {"groups": BASE_GROUPS, "config": cfgp, "trees": npw, "dev_ndcg10": sp,
                                    "search": logp, "test_ndcg10": finish("pointwise", mp, npw, cols)}

    params = {k: cfg[k] for k in ("num_leaves", "min_child_samples")}
    for name, groups in (("lambdamart_ce", BASE_GROUPS + ["ce"]),
                         ("lambdamart_oracle_class", BASE_GROUPS + ["oracle"])):
        c = columns(groups)
        mm, nn, ss = fit(df, c, split["train"], split["dev"], qrels, params, "lambdarank", unj)
        out["variants"][name] = {"groups": groups, "trees": nn, "dev_ndcg10": ss,
                                 "test_ndcg10": finish(name, mm, nn, c)}
        print(name, out["variants"][name], flush=True)

    # leave-one-group-out ablation (same hyperparameters, trees re-chosen on dev)
    for g in BASE_GROUPS:
        groups = [x for x in BASE_GROUPS if x != g]
        c = columns(groups)
        mm, nn, ss = fit(df, c, split["train"], split["dev"], qrels, params, "lambdarank", unj)
        out["variants"][f"ablate_minus_{g}"] = {"groups": groups, "trees": nn, "dev_ndcg10": ss,
                                                "test_ndcg10": finish(f"ablate_minus_{g}", mm, nn, c)}
        print(g, out["variants"][f"ablate_minus_{g}"], flush=True)

    (ROOT / "results" / "ltr.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
