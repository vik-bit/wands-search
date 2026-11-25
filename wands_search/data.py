"""Loading WANDS (Wayfair ANnotation DataSet, MIT licence, github.com/wayfair/WANDS).

Three tab-separated files:
  product.csv  42,994 products (name, class, category hierarchy, description, features, ratings)
  query.csv    480 queries with a query_class
  label.csv    233,448 (query, product) judgements: Exact / Partial / Irrelevant

Decisions made here (each one is explained in docs/LEARNING.md):
  * Gains: Exact = 2, Partial = 1, Irrelevant = 0.
  * 1,575 (query, product) pairs are labelled more than once; 14 of those disagree. We keep the
    highest grade (max), so the gains stay integers, which LightGBM's lambdarank needs.
  * Products that were never judged for a query count as gain 0 at evaluation time. That is the
    usual convention, but it penalises a system that finds relevant products the annotators never
    saw, so the evaluation also reports how many of the top 10 results were judged.
  * Only some of the `product_features` are useful as text. Most of them are compliance and warranty
    fields ("caltb117-2013compliant : no"). The `attrs` field keeps values whose key names a
    colour, material, style, shape, finish, pattern or product type.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

GAIN = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

# Substrings of feature keys whose values go into the `attrs` text field.
ATTR_KEY_PARTS = ("color", "colour", "material", "style", "shape", "finish", "pattern",
                  "producttype", "woodtone", "woodspecies")


def parse_features(raw: str) -> list[tuple[str, str]]:
    """'color : white|overallwidth:64.7' -> [('color', 'white'), ('overallwidth', '64.7')]."""
    out = []
    if not isinstance(raw, str):
        return out
    for part in raw.split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key, value = key.strip().lower(), value.strip()
        if key and value:
            out.append((key, value))
    return out


def attr_text(raw: str) -> str:
    vals = [v for k, v in parse_features(raw) if any(p in k for p in ATTR_KEY_PARTS)]
    # de-duplicate while keeping order ("white", "white" from color and basecolor)
    return " ".join(dict.fromkeys(vals))


def load_products(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "product.csv", sep="\t", encoding="utf-8")
    df = df.rename(columns={"product_name": "name", "product_class": "cls",
                            "category hierarchy": "hierarchy", "product_description": "description",
                            "product_features": "features"})
    for col in ("name", "cls", "hierarchy", "description", "features"):
        df[col] = df[col].fillna("").astype(str)
    df["attrs"] = df["features"].map(attr_text)
    df["rating_missing"] = df["rating_count"].isna().astype(int)
    for col in ("rating_count", "average_rating", "review_count"):
        df[col] = df[col].fillna(0.0)
    return df.sort_values("product_id").reset_index(drop=True)


def load_queries(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "query.csv", sep="\t", encoding="utf-8")
    df["query_class"] = df["query_class"].fillna("")
    return df.sort_values("query_id").reset_index(drop=True)


def load_qrels(data_dir: str | Path) -> dict[int, dict[int, int]]:
    """query_id -> {product_id: gain}. Duplicate judgements resolved by taking the max gain."""
    df = pd.read_csv(Path(data_dir) / "label.csv", sep="\t", encoding="utf-8")
    df["gain"] = df["label"].map(GAIN)
    if df["gain"].isna().any():
        raise ValueError(f"unknown labels: {set(df.loc[df.gain.isna(), 'label'])}")
    df = df.groupby(["query_id", "product_id"], as_index=False)["gain"].max()
    qrels: dict[int, dict[int, int]] = {}
    for qid, pid, g in df.itertuples(index=False):
        qrels.setdefault(int(qid), {})[int(pid)] = int(g)
    return qrels


def make_split(query_ids, seed: int = 0, train: float = 0.5, dev: float = 0.2) -> dict[str, list[int]]:
    """Random split of query ids into train / dev / test (the rest). Deterministic for a seed."""
    ids = sorted(int(q) for q in query_ids)
    random.Random(seed).shuffle(ids)
    n_train, n_dev = round(train * len(ids)), round(dev * len(ids))
    return {"train": sorted(ids[:n_train]),
            "dev": sorted(ids[n_train:n_train + n_dev]),
            "test": sorted(ids[n_train + n_dev:])}


def load_split(path: str | Path) -> dict[str, list[int]]:
    return json.loads(Path(path).read_text())
