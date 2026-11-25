<div align="center">

# wands-search

**A product search relevance study on Wayfair's public WANDS dataset: BM25 with field weights,
dense retrieval on [vecdex](https://github.com/vik-bit/vecdex) (my C++ HNSW library), hybrid
fusion, a cross-encoder reranker, LambdaMART, and LLM query understanding. Every system is
scored on the same 144 held-out queries with graded relevance, 95% bootstrap confidence
intervals and paired significance tests.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-LambdaMART-2c8ebb)
![Compute](https://img.shields.io/badge/compute-laptop%20CPU%20%2B%20free%20Colab%20T4-lightgrey)

</div>

---

> **What this is.** I built this in November 2026 to learn how e-commerce search works, using
> Wayfair's own public data. I had never worked on product search. I wanted to see first-hand
> why lexical matching, embeddings and learned rankers each fail in their own way on short
> furniture queries. It is an exploration on 480 queries, not a production system, and none of
> the numbers below are claims about Wayfair's search. Every number comes from a file in
> [`results/`](results/). Negative results are reported next to the positive ones.
> [`docs/LEARNING.md`](docs/LEARNING.md) has my study notes on every decision.

## At a glance

| | |
|---|---|
| **Data** | [WANDS](https://github.com/wayfair/WANDS) (Chen et al., ECIR 2022, MIT licence): 42,994 products, 480 real queries, 233,448 human judgements (Exact / Partial / Irrelevant). Queries are split once, at random: 240 train / 96 dev / 144 test ([`splits/`](splits/)). |
| **Best result** | LambdaMART on the hybrid top 100: **test NDCG@10 0.805** (95% CI 0.762–0.845), against **0.734** for untuned BM25. The gain is **+0.071** (CI +0.040 to +0.102, p < 0.001). |
| **Biggest single gains** | Tuned field weights in BM25 (+0.035), then adding dense retrieval (+0.015 over tuned BM25), then learning to rank (+0.021, p = 0.061). |
| **Honest negatives** | **LLM query rewriting hurt** (−0.010, p = 0.06), and LLM attribute filters hurt more (−0.039, p = 0.002). Spelling "correction" fixed typos but also changed 10 brand or product-line names. LambdaMART is **no better than a pointwise regressor** on the same features (0.805 vs 0.808). Two off-the-shelf cross-encoders give **no reliable gain**: they hurt on dev and help a little on test, and over the 240 dev + test queries the effect is +0.003 to +0.005 with CIs that include 0. Adding the score as a LambdaMART feature changed nothing. On 43k products, HNSW is **not faster than exact search**. |
| **Checked** | Metrics against hand-computed examples. vecdex against exact search (98.4% of top-10 results identical, NDCG −0.002). Every tuned value is chosen on dev, never on test. |

---

## Results (144 test queries)

Mean with 95% bootstrap CI over queries. NDCG@10 uses graded gains (Exact 2, Partial 1) and is
averaged over 143 queries (one test query has no positive judgement, so its NDCG is undefined). MAP,
recall@50 and MRR count only Exact as relevant, so they are averaged over the 111 test queries
that have at least one Exact product. Full table: [`results/test_results.md`](results/test_results.md).

| system | NDCG@10 | recall@50 | MRR | MAP |
|---|---|---|---|---|
| BM25, all fields concatenated (untuned baseline) | 0.734 (0.684–0.783) | 0.537 | 0.719 | 0.467 |
| BM25, 5 fields, equal weights | 0.728 (0.679–0.778) | 0.508 | 0.649 | 0.391 |
| BM25, 5 fields, weights tuned on dev | 0.769 (0.723–0.815) | 0.509 | 0.699 | 0.454 |
| Dense: bge-small, exact search | 0.745 (0.697–0.792) | 0.514 | 0.697 | 0.440 |
| Dense: bge-small, vecdex HNSW (ef = 200) | 0.743 (0.694–0.790) | 0.513 | 0.696 | 0.440 |
| Hybrid, reciprocal rank fusion (k = 60) | 0.788 (0.742–0.832) | 0.529 | 0.735 | 0.488 |
| Hybrid, weighted min-max fusion (BM25 weight 0.7) | 0.785 (0.739–0.829) | 0.529 | 0.719 | 0.487 |
| + cross-encoder rerank of top 10 (MiniLM-L6, MS MARCO) | 0.793 (0.748–0.836) | 0.529 | 0.761 | 0.503 |
| + cross-encoder rerank of top 10 (bge-reranker-base) | 0.797 (0.753–0.839) | 0.529 | **0.776** | 0.502 |
| + pointwise LightGBM regression | **0.808** (0.765–0.848) | 0.548 | 0.747 | **0.509** |
| + **LambdaMART** | 0.805 (0.762–0.845) | **0.550** | 0.737 | 0.497 |
| + LambdaMART with the cross-encoder score as a feature | 0.802 (0.757–0.843) | 0.550 | 0.739 | 0.500 |
| *LambdaMART + oracle query class (upper bound, not deployable)* | *0.815 (0.774–0.853)* | *0.554* | *0.743* | *0.496* |

**recall@50 cannot reach 1 here.** 41% of test queries have more than 50 Exact products (the
median is 40), so the best possible recall@50 averages 0.785.

### Each step, paired on the same queries

| step | Δ NDCG@10 (95% CI) | p | queries better / worse / same |
|---|---|---|---|
| untuned BM25 → tuned field weights | +0.035 (+0.006 to +0.063) | 0.017 | 62 / 37 / 44 |
| equal field weights → tuned field weights | +0.041 (+0.022 to +0.061) | < 0.001 | 58 / 30 / 55 |
| tuned BM25 → hybrid (weighted) | +0.015 (+0.005 to +0.027) | 0.005 | 44 / 26 / 73 |
| dense only → hybrid (weighted) | +0.042 (+0.008 to +0.076) | 0.023 | 62 / 36 / 45 |
| hybrid → + MiniLM cross-encoder (top 10) | +0.009 (+0.001 to +0.017) | 0.037 | 46 / 24 / 73 |
| hybrid → + bge-reranker-base (top 10) | +0.012 (+0.004 to +0.021) | 0.003 | 50 / 23 / 70 |
| hybrid → LambdaMART | +0.021 (−0.000 to +0.042) | 0.061 | 50 / 33 / 60 |
| pointwise → LambdaMART | −0.003 (−0.016 to +0.011) | 0.70 | 36 / 37 / 70 |
| exact dense → vecdex dense | −0.002 (−0.006 to +0.001) | 0.25 | 2 / 4 / 137 |

p is a two-sided paired randomization test with 10,000 sign flips. About ten comparisons are
made, so p = 0.03 to 0.06 is weak evidence.

### What the numbers say

- **Field weighting is the cheapest large win.** With equal weights the five-field BM25 is
  *worse* than one concatenated field (0.728 vs 0.734). Tuned weights (name ×3, description ×2,
  class / hierarchy / attributes ×1) give +0.041 over equal weights. In every one of the 18 (stem,
  k1, b) settings, the dev search put the name weight at 3 or 5.
- **Stemming mattered more than any BM25 parameter.** With the same tuning, dev NDCG@10 was
  0.789 with a light plural stemmer and 0.722 without ([`bm25_tuning.json`](results/bm25_tuning.json)).
  Queries say "barstools", products say "bar stool"; queries say "lamps", names say "lamp".
- **Dense alone is no better than BM25, but the two fail on different queries.** Dense loses
  badly on queries with a colour (0.590 vs BM25 0.707) or material (0.564 vs 0.736). It wins on
  plain product-type queries (0.819 vs 0.801). Fused, they beat both.
- **Tuning for NDCG traded away Exact-only metrics.** Tuned BM25 has higher NDCG@10 than
  untuned BM25 but lower MRR (0.699 vs 0.719) and MAP (0.454 vs 0.467). NDCG rewards Partial
  matches (the right product type), and the tuning learned to fill the top 10 with them.
- **Off-the-shelf rerankers: the splits disagree.** Both cross-encoders look significant on
  test, but on dev both were *worse* than no reranking at every depth (MiniLM: 0.797 at depth 10,
  0.763 at depth 100, vs 0.803 without). Pooled over all 240 dev + test queries the gains are
  +0.003 (CI −0.003 to +0.010) and +0.005 (−0.002 to +0.011)
  ([`rerank_dev_test_pooled.json`](results/rerank_dev_test_pooled.json)). The consistent signal is
  the depth: the more of the list a web-passage reranker reorders, the worse it gets. It knows
  about relevance in general, but not furniture conventions.
- **LambdaMART's top feature is the hybrid rank itself** (42% of split gain), followed by query-term
  coverage of the name and of all fields (10% each), dense cosine (8%), name BM25 (7%) and the
  class vote (6%) ([`ltr.json`](results/ltr.json)).

### Leave-one-feature-group-out (LambdaMART, test)

| removed group | test NDCG@10 | Δ vs full model (95% CI) | p |
|---|---|---|---|
| query-term coverage | 0.794 | −0.011 (−0.026 to +0.004) | 0.16 |
| class vote + class-in-query | 0.806 | +0.000 (−0.010 to +0.011) | 0.94 |
| review / rating stats | 0.806 | +0.001 (−0.017 to +0.018) | 0.94 |
| query / name length | 0.806 | +0.001 (−0.017 to +0.018) | 0.94 |
| hybrid rank | 0.810 | +0.005 (−0.005 to +0.015) | 0.36 |
| dense cosine | 0.812 | +0.007 (−0.008 to +0.023) | 0.41 |
| per-field BM25 | 0.816 | +0.011 (+0.003 to +0.019) | 0.011 |

The "reviews" and "query length" rows are identical, and that is not a bug. Without either
group, dev NDCG peaked after 25 trees, and neither group is used in the full model's first 25
trees. So both runs are exactly the full model cut at 25 trees. Review stats only enter the
later trees. Removing them makes the model stop earlier, but test NDCG barely moves.

No single group is necessary; the features overlap heavily (the hybrid rank already contains the
BM25 and dense scores). Removing the per-field BM25 scores *improved* test NDCG with p = 0.011, but
not on dev (0.817 vs 0.820). With seven ablations, I read this as "the BM25 features are
redundant given the rank", not as "BM25 features hurt".

---

## LLM query understanding

One call per query to **Qwen2.5-7B-Instruct (4-bit)**, run on a free Colab T4 GPU with
[`notebooks/llm_query_understanding_kaggle.ipynb`](notebooks/llm_query_understanding_kaggle.ipynb)
at about 0.9 s per query. It returns JSON with a spelling-corrected query, the product type,
colour, material, size, style and a catalogue-style rewrite. All 480 outputs parsed, and they are
committed in [`results/llm/`](results/llm/). Each variant changes only the query side and reuses
the same first stage (tuned BM25 + bge-small, weighted fusion). Nothing is trained, and the one
tuned value (the boost weight) is chosen on dev ([`llm_qwen2.5-7b-instruct-4bit.json`](results/llm_qwen2.5-7b-instruct-4bit.json)).

| variant (test, 144 queries) | NDCG@10 | Δ vs original query (95% CI) | p | better / worse / same |
|---|---|---|---|---|
| original query (= weighted hybrid)¹ | 0.784 | — | — | — |
| spelling-corrected query | 0.785 | +0.001 (−0.003 to +0.004) | 0.75 | 2 / 1 / 140 |
| LLM rewrite instead of the query | 0.774 | −0.010 (−0.021 to −0.000) | 0.059 | 14 / 18 / 111 |
| original + rewrite, fused (RRF) | 0.780 | −0.005 (−0.015 to +0.004) | 0.39 | 14 / 13 / 116 |
| boost products with the extracted colour / material | 0.784 | 0 (dev chose a boost weight of **0**) | — | — |
| **filter** to products with the extracted colour / material | 0.746 | **−0.039** (−0.067 to −0.014) | 0.002 | 5 / 13 / 125 |

¹ 0.784 here vs 0.785 in the main table. The script rebuilds the vecdex index, and the most likely
cause is that a multi-threaded HNSW build is not bit-for-bit repeatable, so a few dense lists differ
slightly. I did not trace it further.

**Where it helped:** real typos. "foutains with brick look" → "fountains" (NDCG 0.24 → 0.42),
"glass lsmp shades" → "lamp" (0.47 → 0.69, dev), "pedistole sink" → "pedestal" (0 → 0.07).

**Where it hurt:**

- **Brand and product-line names get "corrected".** The prompt says not to change names the model
  does not recognise, but of the 26 real spelling changes over all 480 queries, 10 alter a brand or
  product-line name, 1 is plainly wrong ("shirt head" → "shirthead"), and 15 fix a real typo or format:
  "serta kleinmon" → "telsey kleinmon" (dev NDCG 0.94 → 0.49), "regner power loom red" → "regal"
  (1.00 → 0.82), "orren ellis" → "Oren Ellis" (Orren Ellis is a Wayfair house brand),
  "mahone porch rocking chair" → "mahogany". In a catalogue full of invented product names,
  "unfamiliar word" usually means "brand", not "typo".
- **Rewrites lose the customer's words.** "desk for kids" → "desk suitable for children" (0.65 → 0.22):
  product names say "kids", not "children". "70s inspired furniture" → "70s style furniture"
  (0.33 → 0.10). The rewrite looks more "catalogue-like" to the LLM but matches this catalogue worse.
- **Hard attribute filters throw away the near misses.** "gray leather cocktail table" went from
  0.93 to 0, and "olive green console table" from 0.92 to 0. Neither query has a single Exact product
  in the catalogue: the best available are Partials (leather cocktail tables that aren't gray, other
  console tables). The filter keeps only products whose name or attributes contain every extracted
  word, so it drops all of them. Only 4 of 100 results survived for the first query, all of them
  irrelevant, and 0 of 100 for the second. A ranker that prefers attribute matches can degrade
  gracefully; a filter cannot. Extraction also reads colours out of brand names: "Kohler
  Whitehaven" → white, "Palram Harmony greenhouses" → green.

**Takeaway.** Used naively, as a replacement for the query, the LLM is neutral to harmful here.
What I would try next is spelling correction gated on the catalogue vocabulary (only "correct" a
word that is not in any product name, brand or class), and attributes used as *features* for the
ranker rather than as filters. The 4-bit model and the single prompt are limitations too: I did not
try a larger model or prompt variants.

---

## Error analysis (test queries)

NDCG@10 by rule-based query type ([`error_analysis.md`](results/error_analysis.md); a query can
have several types; groups under 20 queries are noisy):

| query type | n | tuned BM25 | dense | hybrid RRF | + rerank | LambdaMART |
|---|---|---|---|---|---|---|
| plain product type ("shoe closet") | 74 | 0.801 | 0.819 | 0.831 | 0.825 | 0.846 |
| has an attribute (colour, material, style, size) | 50 | 0.734 | 0.673 | 0.753 | 0.767 | 0.768 |
| has a colour | 12 | 0.707 | 0.590 | 0.739 | 0.756 | 0.774 |
| has a material | 9 | 0.736 | 0.564 | 0.709 | 0.824 | 0.767 |
| has a size or number | 20 | 0.696 | 0.694 | 0.723 | 0.706 | 0.702 |
| has a rare word (brand, product line, typo) | 24 | 0.754 | 0.665 | 0.741 | 0.762 | 0.769 |
| 1–2 words | 45 | 0.749 | 0.761 | 0.776 | 0.780 | 0.792 |
| 5+ words | 16 | 0.769 | 0.719 | 0.778 | 0.787 | 0.798 |

What goes wrong, from the worst queries:

- **Sizes and numbers are the weakest group for every system** (≈0.70). Tokenised, "30 inch"
  is just "30", which matches dimensions, model numbers and pack sizes alike.
- **Misspellings sink every stage.** "pedistole sink" never matches "pedestal", so only "sink"
  matches and the top five are vessel sinks. "foutains with brick look" has the same problem.
- **Wrong head noun.** "outdoor waterproof chest" (deck boxes) returns waterproof *covers*,
  because "waterproof" and "outdoor" match many cover products strongly.
- **Unjudged results.** For "living room ideas" the top five are 3-piece living room sets, which
  look relevant, but none of them were judged, so NDCG@10 = 0. Unjudged products count as
  irrelevant. 8–17% of every system's top 10 is unjudged (judged@10 ranges from 0.83 to 0.92),
  so part of every gap in the table is about which products the annotators happened to see.

---

## Latency (per query, one at a time)

Laptop CPU (Intel i7-1255U, WSL2), models warm, 144 test queries ([`latency.json`](results/latency.json)):

| stage | p50 | p95 |
|---|---|---|
| BM25, 5 fields, 43k products (scipy sparse) | 17 ms | 47 ms |
| encode the query (bge-small, ONNX) | 22 ms | 63 ms |
| vecdex HNSW search, top 100 | 10 ms | 104 ms |
| exact search, top 100 (numpy, 43k × 384) | 21 ms | 42 ms |
| fusion | 0.3 ms | 1.6 ms |
| LambdaMART: features + predict for 100 candidates | 100 ms | 171 ms |
| **BM25 + dense + fusion + LambdaMART** | **167 ms** | **322 ms** |
| cross-encoder on 50 candidates (for comparison) | 1,008 ms | 1,654 ms |

- **At 43k products, exact search is as fast as HNSW.** A 43k × 384 matrix product takes ~20 ms.
  The approximate index only pays off with millions of products. Here it only shows that the
  ANN path loses almost nothing (2 queries better, 4 worse, 137 identical).
- **The LambdaMART stage costs 100 ms**, measured as features + prediction together. I did not
  time the two separately. The feature code is plain Python looping over the 100 candidates, and
  that is where I would look first.
- The MiniLM cross-encoder costs ~6× the whole LambdaMART pipeline and was not clearly better. bge-reranker-base takes 2.4 s for just 20 pairs.

---

## What I would do with real click data

WANDS has human judgements only. A real search team has far more behaviour data than labels,
and it changes the method:

1. **Clicks are biased by position.** Users click the first results partly because they are
   first. Training on raw clicks teaches the ranker to repeat the old ranking. Standard fixes:
   estimate the examination probability per position (randomise the top few results for a small
   share of traffic, or use intervention harvesting from past ranker changes), then train with
   inverse propensity weighting, or fit a click model (cascade, DBN) that separates "seen" from
   "relevant".
2. **Use the right label for the goal.** For e-commerce, add-to-cart and purchase are stronger
   relevance signals than clicks, but rarer and also biased (price, delivery date). I would
   train on a graded label such as purchase > add-to-cart > long click > click > skip, and check
   it against human labels like WANDS on a sample.
3. **Offline metrics only pick candidates; online A/B tests decide.** Online, the metrics are
   search conversion rate, revenue per search, click-through rate, null-result and reformulation
   rate, and time to first click. Before an A/B test, **interleaving** (mixing two rankers' results
   in one list) detects preferences with far fewer users.
4. **Check offline–online agreement.** This study found two systems (LambdaMART and pointwise) that
   offline metrics cannot separate. Only online data can say whether they differ for customers.

---

## What was verified

| claim | how it is checked |
|---|---|
| NDCG, AP, recall, RR, judged@k are correct | [`tests/test_metrics.py`](tests/test_metrics.py): hand-computed worked examples (NDCG@4 = 3.361 / 4.693 = 0.716319), undefined-query handling |
| bootstrap and randomization test behave | [`tests/test_stats.py`](tests/test_stats.py): exact enumeration on 3 queries (p = 2/8), a real shift detected, noise not, CI width ≈ 1.96·σ/√n |
| BM25 weights | [`tests/test_bm25.py`](tests/test_bm25.py): one term weight worked by hand (idf = ln 2, tf part = 0.88) |
| data loading | [`tests/test_data.py`](tests/test_data.py) on a 4-product fixture: duplicate labels → max, missing fields, attribute extraction |
| fusion | [`tests/test_fusion.py`](tests/test_fusion.py): RRF and min-max fusion by hand, deterministic ties |
| vecdex ≈ exact | in CI on random clustered vectors (> 95% top-10 overlap); on WANDS, top-10 identical on 442 / 480 queries at ef = 200 ([`dense_check.json`](results/dense_check.json)) |
| no test leakage | BM25 weights, k1, b, stemming, fusion weight, rerank depth, LightGBM settings and tree count are all chosen on dev ([`bm25_tuning.json`](results/bm25_tuning.json), [`fusion_tuning.json`](results/fusion_tuning.json), [`rerank_MiniLM.json`](results/rerank_MiniLM.json), [`ltr.json`](results/ltr.json)) |

`pytest -q` runs 31 tests in ~10 s on CPU; CI runs them on Python 3.10 and 3.12 with vecdex
built from source.

## What was not measured

- **One random split, one seed.** With 144 test queries, CIs are about ±0.04 NDCG. I did not
  repeat the study over several splits, so the ranking of close systems (pointwise vs
  LambdaMART, RRF vs weighted) is not established.
- **No fine-tuning.** The embedding model and the cross-encoder are used off the shelf. Fine-tuning
  bge-small or the cross-encoder on the 240 training queries is the obvious next step and would
  probably change the cross-encoder result. I did not do it.
- **No product images** and no use of the long descriptions in the dense model.
- **Pool bias.** WANDS' candidates were pooled for annotation in a way I can't see. Systems that
  surface different products from the pool get no credit for them.
- **Not Wayfair's traffic.** The 480 queries are not weighted by how often customers search for
  them, so the head queries that matter most commercially count the same as rare ones.
- **Latency** is on a laptop under WSL2, one query at a time, with no batching or caching. It
  varied by up to 2× between runs and is only good for comparing stages with each other.
- **The demo** reproduces the hybrid + optional rerank pipeline, not LambdaMART or the LLM step.
- **The LLM step** used one model (Qwen2.5-7B, 4-bit), one prompt and temperature 0, on dev + test queries only.

---

## Demo

```bash
streamlit run app/demo.py
```

Type a query and see the ranked products with their BM25, dense, fused and (optionally)
cross-encoder scores, plus the WANDS label when the query is one of the 480.

## Reproduce

Everything except the LLM step runs on a laptop CPU (the product embeddings took 29 minutes).

```bash
bash scripts/download_data.sh                      # WANDS CSVs into data/raw/
pip install -e ".[models,ann,demo,dev]"
python scripts/run_bm25.py                         # BM25 tuning on dev
python scripts/run_dense.py                        # embeddings, vecdex vs exact
python scripts/run_fusion.py
python scripts/run_rerank.py                       # cross-encoder on the hybrid top 100
python scripts/run_rerank.py BAAI/bge-reranker-base --dev-test --max-depth=20
python scripts/run_ltr.py                          # LambdaMART, pointwise, ablations
python scripts/evaluate.py && python scripts/ablation_table.py && python scripts/error_analysis.py
python scripts/latency.py
python scripts/run_llm.py --backend cache --model qwen2.5-7b-instruct-4bit   # uses the committed LLM outputs
```

Use `PYTHONPATH=.` if the package is not installed. The LLM outputs come from
[`notebooks/llm_query_understanding_kaggle.ipynb`](notebooks/llm_query_understanding_kaggle.ipynb)
on a free GPU (I ran it on Colab; it also runs on Kaggle). They are committed in [`results/llm/`](results/llm/), so the analysis
reruns without a GPU.

## Layout

```
wands_search/   data.py metrics.py stats.py text.py bm25.py dense.py fusion.py rerank.py ltr.py llm.py query_tags.py runs.py
scripts/        one script per stage (above)
results/        every number in this README; results/runs/ has the top 100 of every system for every query
splits/         the fixed train / dev / test query split
docs/           LEARNING.md: study notes, worked examples, decisions and the alternatives rejected
app/            Streamlit demo
tests/          31 CPU tests, including a 4-product fixture
```

## Data and licence

WANDS is released by Wayfair under the MIT licence (copyright 2021, the ECIR 2022 dataset
authors; [github.com/wayfair/WANDS](https://github.com/wayfair/WANDS)). It is downloaded by
`scripts/download_data.sh`, not copied into this repository. Models: BAAI/bge-small-en-v1.5
(MIT), ms-marco-MiniLM-L-6-v2 (Apache-2.0), BAAI/bge-reranker-base (MIT), Qwen2.5-7B-Instruct
(Apache-2.0). This project's code is MIT.
