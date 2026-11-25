"""LLM query understanding: spelling correction, attribute extraction and rewriting.

One call per query returns JSON. Two free backends, both through plain HTTP (no SDK):
  groq    https://api.groq.com/openai/v1 (OpenAI-compatible), key from $GROQ_API_KEY
  ollama  http://localhost:11434 (local model, CPU)
Every response is cached in results/llm/<model>.jsonl and committed, so the analysis reruns
without a key and the exact outputs behind every number can be read.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

PROMPT = """You help a furniture and home-goods store understand search queries.
For the query below, return ONLY a JSON object with these keys:
  "corrected": the query with spelling mistakes fixed; identical to the input if there are none.
               Do not change brand or product-line names you do not recognise.
  "product_type": the kind of product being searched for, as a short noun phrase, or null
  "color": colour words in the query, or null
  "material": material words in the query, or null
  "size": size or dimension words in the query (e.g. "queen", "24 inch", "small"), or null
  "style": style words in the query (e.g. "modern", "farmhouse"), or null
  "rewrite": the query rewritten as a store's catalogue would describe the product. Keep every
             constraint in the query, do not add new ones, at most 10 words.
Only use words that are in the query for color, material, size and style; do not guess.

Query: {query}"""

KEYS = ("corrected", "product_type", "color", "material", "size", "style", "rewrite")


def _post(url: str, payload: dict, headers: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call(query: str, backend: str, model: str) -> str:
    prompt = PROMPT.format(query=query)
    if backend == "groq":
        body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}
        r = _post("https://api.groq.com/openai/v1/chat/completions", body,
                  {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                   "User-Agent": "wands-search-study"})
        return r["choices"][0]["message"]["content"]
    if backend == "ollama":
        body = {"model": model, "prompt": prompt, "format": "json", "stream": False,
                "options": {"temperature": 0, "seed": 0}}
        return _post("http://localhost:11434/api/generate", body, {})["response"]
    raise ValueError(backend)


def parse(raw: str, query: str) -> dict:
    """Tolerant JSON parse; missing or malformed fields fall back to the original query."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(m.group(0)) if m else {}
    out = {}
    for k in KEYS:
        v = obj.get(k)
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        out[k] = str(v).strip() if v not in (None, "", "null", "None") else None
    out["corrected"] = out["corrected"] or query
    out["rewrite"] = out["rewrite"] or query
    out["parse_ok"] = bool(obj)
    return out


def run_all(queries: dict, backend: str, model: str, cache_path: Path, sleep: float = 0.0) -> dict:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["query_id"]] = r
    with cache_path.open("a", encoding="utf-8") as fh:
        for qid, q in queries.items():
            if qid in done:
                continue
            for attempt in range(5):
                try:
                    t0 = time.perf_counter()
                    raw = call(q, backend, model)
                    sec = time.perf_counter() - t0
                    break
                except Exception as e:  # rate limits on the free tier: back off and retry
                    print("retry", qid, e, flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"giving up on query {qid}")
            rec = {"query_id": qid, "query": q, "raw": raw, "seconds": round(sec, 3)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[qid] = rec
            time.sleep(sleep)
    return {qid: {**parse(r["raw"], r["query"]), "seconds": r["seconds"]} for qid, r in done.items()}
