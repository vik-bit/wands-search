# Learning notes: wands-search

These notes are for me, to study before talking about this project. Every number quoted here is
in a file under `results/`; when this file and a result file disagree, the result file is right.

---

## 1. The problem, in Wayfair's terms

A customer types a short query ("tall buffet cabinet", "nectar queen mattress"). The search
system must pick, out of ~43k products in this dataset (tens of millions at Wayfair), the few
dozen to show, in order. Two stages are standard:

1. **Candidate generation (recall).** Cheap methods (an inverted index for BM25, an ANN index for
   embeddings) pull a few hundred plausible products out of the whole catalogue. A product that is
   not retrieved here can never be shown, so the metric for this stage is **recall@50/100**.
2. **Ranking (precision at the top).** A more expensive model reorders the candidates. Customers
   look at the first screen, so the metric is **NDCG@10**.

WANDS gives human judgements for 480 real queries: each judged (query, product) pair is **Exact**
(the product is what was asked for), **Partial** (related but misses a constraint: right type,
wrong colour; or a matching accessory) or **Irrelevant**. On average 486 products are judged per
query, but the catalogue has 42,994, so most products are never judged for a given query.

---

## 2. How NDCG@10 is computed (worked example)

Gains: Exact = 2, Partial = 1, Irrelevant = 0, unjudged = 0.

**DCG@k** sums each result's gain, discounted by its position:

    DCG@k = sum over positions i = 1..k of  gain_i / log2(i + 1)

The discount is 1 at position 1, 0.63 at 2, 0.5 at 3, 0.43 at 4, 0.30 at 10. A relevant product at
position 1 is worth 3.3 times as much as the same product at position 10.

**NDCG@k** = DCG@k / IDCG@k, where IDCG is the DCG of the best possible ordering of *all* the
products judged for this query, not only the ones the system returned. So NDCG is 1 only if the
system put the best judged products at the top, and it is between 0 and 1 for every query, which
makes queries comparable however many relevant products they have.

Worked example (this is `tests/test_metrics.py`). Judged: a, d, e are Exact, c is Partial, b is
Irrelevant. The system returns a, x, c, d (x was never judged).

| position i | product | gain | discount 1/log2(i+1) | gain × discount |
|---|---|---|---|---|
| 1 | a | 2 | 1.000 | 2.000 |
| 2 | x (unjudged) | 0 | 0.631 | 0 |
| 3 | c | 1 | 0.500 | 0.500 |
| 4 | d | 2 | 0.431 | 0.861 |
| | | | **DCG@4** | **3.361** |

Ideal order of all judged gains: 2, 2, 2, 1, 0 → IDCG@4 = 2 + 2(0.631) + 2(0.5) + 1(0.431) = 4.693.
**NDCG@4 = 3.361 / 4.693 = 0.716.**

Things to be able to say about it:
- **Linear vs exponential gain.** Some papers use gain = 2^rel − 1 (Exact = 3, Partial = 1), which
  makes Exact count 3× a Partial instead of 2×. I used linear gains, and told LightGBM to use the same
  gains (`label_gain=[0, 1, 2]`). Its default, [0, 1, 3], is the exponential one, so without that
  setting the model would be trained for a slightly different metric than the one it is scored on.
- **Undefined queries.** If a query has no positive judgement at all, IDCG = 0 and NDCG is
  undefined. I drop it from the mean instead of scoring it 0 (one test query is like this).
- **Unjudged = 0 is a bias.** A system that finds good products the annotators never saw is
  scored as if they were irrelevant. So I also report **judged@10**, the share of the top 10
  that has a label. If two systems differ a lot in judged@10, their NDCG gap is partly an
  artefact of how the labels were collected.
- **MAP, recall@50 and MRR are binary**: only Exact counts as relevant. 101 of 480 queries have no
  Exact product, so those metrics are averaged over fewer queries (the `n` is in the result files).

---

## 3. Statistics: why paired, why bootstrap

Two systems are compared on the same queries. Per-query NDCG varies hugely: some queries are easy
for every system. If I compared two means with an unpaired test, that query-to-query variance
would swamp a real 0.01 difference. Looking at the *per-query difference* B − A removes it.

- **Paired bootstrap CI**: resample the 144 test queries with replacement 10,000 times; each time
  take the mean of B − A; the 2.5th and 97.5th percentiles are the 95% CI. If it excludes 0, the
  improvement is unlikely to be an accident of which queries ended up in the test set.
- **Randomization (permutation) test**: if A and B were really equally good, then for each query
  it would be arbitrary which one got the higher score, so flipping the sign of each difference
  at random gives the null distribution. p = share of 10,000 random sign patterns whose |mean| is
  at least as large as the observed one.
- **What they don't cover**: they treat the test queries as a random sample of "queries like
  these". They say nothing about queries unlike WANDS' 480 (e.g. Wayfair's real traffic
  distribution, which is head-heavy), or about label noise.
- **Multiple comparisons**: I compare many pairs. With ~10 comparisons at p < 0.05, about one
  "significant" result could be luck, so a p of 0.03 here is weak evidence.

---

## 4. Why learning to rank (LambdaMART) instead of a pointwise classifier

**Pointwise**: treat each (query, product) row on its own and predict its label (regression on
the gain, or classification Exact vs not). The loss is the same for an error at position 1 and at
position 90, and the same for every query. But the model is only ever used to *order products
within one query*. Two problems follow:

1. **Scores are not comparable across queries, and need not be.** A pointwise model spends effort
   making "0.7 for the sofa query" mean the same thing as "0.7 for the lamp query", which ranking
   never needs.
2. **It optimises the wrong thing.** Getting the order of positions 1–3 right matters far more for
   NDCG@10 than positions 50–60, but squared error counts them all the same.

**Pairwise (RankNet)**: the loss is over pairs of products within the same query: for every pair
where i is more relevant than j, push s_i − s_j up (logistic loss on the difference). That
fixes problem 1 but not problem 2: all misordered pairs cost the same.

**LambdaRank / LambdaMART**: take RankNet's gradient for each pair and multiply it by |ΔNDCG|, the
change in NDCG if i and j swapped places. A swap between positions 1 and 2 changes NDCG a lot, a
swap between 80 and 81 almost not at all, so the model concentrates on the top of the list. These
weighted pair gradients (the "lambdas") are then fitted by gradient-boosted trees (MART). The
result empirically optimises NDCG even though NDCG itself is not differentiable.

Tree models also suit these features: BM25 scores and review counts are on wildly different
scales and have thresholds and interactions ("high BM25 on name matters only if the class also
matches"); trees need no normalisation and find interactions themselves.

What I measured: see section 7 — I trained a pointwise regressor with the *same* features, same
hyperparameter search and same early stopping, so the objective is the only difference.

---

## 5. How each feature was chosen

Candidates are the weighted hybrid's top 100 per query. Each row is one (query, product) pair.
Share of split gain in the trained model (`results/ltr.json`) is in brackets.

| feature | why it is there | what it ended up doing |
|---|---|---|
| `hybrid_rank` | The first stage already combines BM25 and dense well (0.785). Giving the model the rank lets it start from that ordering and learn corrections instead of re-deriving it. | Top feature (42%). Most of what LambdaMART does is "keep the hybrid order unless something else disagrees strongly". |
| `bm25_name`, `_cls`, `_hierarchy`, `_attrs`, `_description`, `bm25_total` | Fixed field weights force one trade-off for every query. Per-field scores let the trees learn interactions, e.g. "a description-only match is fine if the class also matches". | Name BM25 is the 5th most used (7%), the rest are small. Removing the whole group did not hurt (the ablation even improved test, but not dev). |
| `dense_cos` | Semantic match, including synonyms ("couch" vs "sofa") that BM25 misses. | 8%. |
| `cover_name`, `cover_any` | BM25 rewards repeated common words. Coverage asks a different question: *how many of the query's words* does the product contain? A product missing "outdoor" in "outdoor rug" is usually Partial, however high its BM25. | 2nd and 3rd (about 10% each). The only group whose removal lowered test NDCG (−0.011, not significant). |
| `class_vote` | Share of the top 20 candidates with this product's class. It is a cheap guess at the query's intended category with no extra model: if 14 of the top 20 are "Bar Stools", a "Kitchen Island" in the list is suspicious. | 6%. |
| `class_in_query` | Share of the class name's words that appear in the query ("accent chair" vs class "Accent Chairs"). | Barely used. |
| `log_reviews`, `log_ratings`, `avg_rating`, `rating_missing` | Popularity is a classic e-commerce prior: among equally relevant products, popular ones are more often what people want. Log-scaled because counts are heavy-tailed. | Not used in the first 25 trees, and removing them changed nothing measurable. **WANDS labels are relevance, not purchase**, so a popularity feature has nothing to predict here. With click or purchase labels it would matter. |
| `query_len`, `name_len` | Query-level context lets trees treat short head queries ("sofa") differently from long specific ones. | Not measurably useful. |
| `ce_score` (variant) | Cross-encoder relevance. | Adding it changed nothing (−0.004, p = 0.51). |
| `oracle_class_match` (upper bound) | 1 if the product's class equals the annotated query class. **Not deployable**: a real system would need a query classifier. It shows how much a perfect classifier could add. | +0.010 (p = 0.23). Even perfect category knowledge adds little on top of `class_vote`. |

Labels: gains 0/1/2 with `label_gain=[0, 1, 2]`. Unjudged candidates are labelled 0. The
alternative, dropping them from training, scored worse on dev (0.795 vs 0.820). See surprise 3.

---

## 6. Design decisions and the alternative I rejected

| decision | alternative rejected | why |
|---|---|---|
| Split by **query** (240/96/144), fixed in `splits/` | Split by (query, product) row | Rows from one query in both train and test would leak that query's words and class into training. Real search meets *new queries*. |
| Every tuned value chosen on **dev**; test used only for the final numbers | Tune on test, or report the best of many test runs | Otherwise the test number is an overestimate. All dev choices are in `results/*_tuning.json`, `rerank_MiniLM.json` and `ltr.json`. |
| Duplicate labels → **max** grade | Mean grade; most recent label | Keeps gains integers (LightGBM's lambdarank needs integer labels). Only 14 pairs disagree, so it barely matters. |
| Unjudged = **gain 0**, plus a `judged@10` column | Score only judged products ("condensed lists") | Condensed lists hide the problem of retrieving unjudged products. judged@10 shows how big it is (8–17% of the top 10). |
| Graded **NDCG@10** as the main metric | MAP / MRR | WANDS has three grades, and a Partial (right type, wrong colour) is better than an Irrelevant result. MAP and MRR need binary labels and throw that away. They are still reported, with Exact as relevant. |
| **Weighted sum of per-field BM25** | BM25F (combine weighted term frequencies before saturation) | BM25F is more principled, but the sum gives a per-field score LambdaMART can use, and it is what Elasticsearch's `most_fields` does. BM25F is a reasonable next experiment. |
| **Light plural stemmer** (hand-written, about 20 lines) | Porter stemmer; no stemming | No stemming cost 0.067 dev NDCG. Porter over-conflates catalogue words ("organic" and "organ" both become "organ") and is harder to explain. |
| **`attrs` field** = colour / material / style / shape / finish / pattern values only | All feature values | Most features are compliance and warranty fields ("caltb117-2013compliant : no"); their tokens ("no", "yes", "compliant") are noise to BM25. |
| **Dense text** = name + class + attributes, no description | Include the description | Descriptions are long marketing copy. Transformer cost grows with input length, so they would make the 29-minute CPU encoding several times longer, and they pull the embedding towards generic phrases. Not measured: I didn't run the with-description variant. |
| **bge-small-en-v1.5** (33M, 384-d) via ONNX | bge-base / e5-large; paid embedding APIs | It runs on a laptop CPU, is free and open, and is strong for its size. A bigger model is a clean next experiment. |
| **vecdex** HNSW, checked against exact search | FAISS / hnswlib | It is my own library, and checking it against exact search on real data is itself a result (442 / 480 identical top 10s, NDCG −0.002). At this scale exact search is just as fast, which I say openly. |
| **Weighted min-max fusion** as the candidate list | RRF | Weighted won on dev (0.803 vs 0.795). On test they are tied (−0.004, p = 0.71). RRF needs no tuning and is the safer default. I would pick RRF in production unless an online test said otherwise. |
| Cross-encoder **off the shelf** | Fine-tune it on the 240 training queries | Fine-tuning is the obvious fix for the domain gap, but I wanted the baseline first. Not done, and listed as not measured. |
| **LambdaMART** with small trees (7 leaves), trees chosen on dev | Bigger trees; a neural ranker | Only 240 training queries (about 24k rows). Bigger trees scored lower on dev (31 leaves: 0.807 vs 0.820). |
| **Pointwise regression** trained with the *same* search | Compare against an untuned pointwise model | Otherwise "LTR beats pointwise" could just mean "the tuned model beats the untuned one". |
| **Paired bootstrap + randomization test** over queries | Unpaired t-test; no CIs | Pairing removes query difficulty, the largest source of variance (section 3). |
| LLM: one call per query, cached, temperature 0 | Calling the LLM at query time | Cached outputs make every number reproducible without a GPU or key. In production an LLM would need its own cache anyway, because head queries repeat. |
| LLM on **dev + test queries only** | All 480 | Nothing in the LLM step is trained; its one tuned value (the boost weight) is chosen on dev. |

---

## 7. Three surprising results

1. **LambdaMART did not beat pointwise regression** (test 0.805 vs 0.808; −0.003, CI −0.016 to
   +0.011). Section 4's theory predicts the opposite. My reading of why:
   - The candidates arrive already well ordered, and the top feature is the hybrid rank. Both
     models mostly learn "keep the order", which leaves the objective little to decide.
   - WANDS labels are dense and graded (about 486 judged per query), so regressing the gain is
     already a decent stand-in for ordering.
   - With 144 test queries the CI is ±0.013, so a small real difference would not show.

   What I *can* say is that the features and the first stage mattered much more than the
   objective on this data. Listwise objectives usually show their advantage with sparser, noisier
   labels such as clicks, where only the top matters.

2. **Field weights mattered more than adding a neural model, and equal weights are worse than one
   concatenated field.** Five equally weighted fields score 0.728 and one concatenated field 0.734.
   The description is long, so its BM25 scores are large and swamp the name. Tuned weights gave
   +0.041, more than adding dense retrieval to tuned BM25 (+0.015), and stemming alone was worth
   0.067 on dev. Cheap text-processing decisions came first in this study.

3. **Treating unjudged products as irrelevant during training helped a lot** (dev 0.820 vs 0.795
   when they are dropped). The comfortable explanation is that most unjudged products really are
   irrelevant. The uncomfortable one is that the model learns to **imitate the annotation pool**:
   it pushes up products that look like what was judged, which is exactly what "unjudged = 0"
   rewards at evaluation. Both are probably true. It is the same pool bias the error analysis
   shows, where "living room ideas" scores 0 with five plausible but unjudged results. Click data
   has the same trap: you only learn from what was shown.

Also worth knowing: **both off-the-shelf cross-encoders hurt on dev and helped on test.** MiniLM
was 0.797 vs 0.803 on dev and +0.009 (p = 0.037) on test. bge-reranker-base was 0.796 on dev and
+0.012 (p = 0.003) on test. Pooled over all 240 dev + test queries, neither gain is distinguishable
from zero (+0.003 and +0.005). That flip is why one split should not be trusted for small effects,
and why the README reports the pooled number. The one consistent signal is depth: reordering more
of the list makes things worse (MiniLM on dev: 0.797 at 10, 0.791 at 20, 0.778 at 50, 0.763 at 100).
A web-passage reranker doesn't know furniture conventions: "outdoor waterproof chest" means deck
boxes, not covers.

---

## 8. Questions I should be able to answer

- *Why NDCG and not accuracy?* Ranking quality is about the order at the top, with graded
  relevance. Accuracy ignores position.
- *Why is IDCG over all judged products, not only the retrieved ones?* So a system can't score
  1 by returning one mediocre product. Otherwise recall failures would be invisible.
- *How would you choose between two systems the CIs can't separate?* An online interleaving or
  A/B test on conversion. Offline: more queries, or several splits.
- *What would you change with Wayfair's data?* Click and purchase labels with position-bias
  correction (see the README). Fine-tune the embedding model and cross-encoder on in-domain
  pairs. Add a query classifier: the oracle says perfect category knowledge is worth only ~0.01
  here, but it is cheap. Add images.
- *Why does dense lose on colour and material queries?* One 384-d vector for "black metal bed" is
  dominated by "bed"; the attribute words move it only a little. BM25 rewards each matching word.
- *Why is exact search as fast as HNSW here?* 43k × 384 floats is 66 MB, and one matrix–vector
  product over it takes ~20 ms. HNSW's advantage grows with the corpus (roughly log n vs n).
- *Is the +0.071 over BM25 real?* 95% CI +0.040 to +0.102 on 144 queries, p < 0.001, and every
  tuned value was chosen on dev. It is real for queries like WANDS'. Whether it holds for
  Wayfair's traffic mix is unknown.

---

## 9. The LLM step: what to say about it

**Setup.** Qwen2.5-7B-Instruct in 4-bit on a free Colab T4, one JSON call per query
(corrected query, product type, colour, material, size, style, rewrite), temperature 0, outputs
cached. Each variant changes only the query side; the first stage is unchanged.

**Result: neutral to harmful, and the failure modes are the interesting part.**

| variant | test Δ NDCG@10 | why |
|---|---|---|
| spelling correction | +0.001 (n.s.) | Only 5 test queries change. It fixes real typos ("foutains" → "fountains", +0.18) but "corrects" brand names. Of the 26 real changes over 480 queries, 10 alter a brand or product-line name ("serta kleinmon" → "telsey kleinmon": 0.94 → 0.49; "orren ellis", a Wayfair house brand, → "Oren Ellis") |
| rewrite instead of query | −0.010 (p = 0.059) | Rewrites swap the customer's words for "nicer" ones the catalogue doesn't use: "desk for kids" → "desk suitable for children" (0.65 → 0.22) |
| original + rewrite (fused) | −0.005 (n.s.) | Expansion is safer than replacement, but still not a gain |
| attribute boost | 0 | Every boost weight > 0 made dev worse, so dev picked 0 |
| attribute filter | −0.039 (p = 0.002) | The worst cases have **no** Exact product in the catalogue ("olive green console table"). The best available results are Partials that miss one constraint, and a filter on every extracted word removes exactly those (0 of 100 results kept). Extraction also finds colours inside brand names ("Kohler Whitehaven" → white) |

**Why this matters for e-commerce.** A catalogue is full of invented proper nouns (brands,
collections, product names). A general LLM has never seen most of them, so "unfamiliar" looks like
"misspelled" to it. Query understanding has to be **grounded in the catalogue**:
- only correct a word that appears in no product name, brand or class;
- prefer corrections that exist in the catalogue vocabulary;
- use extracted attributes as **soft features** for the ranker (does the product's colour field
  match?), never as hard filters, unless the attribute is reliably structured;
- measure on real traffic: in WANDS the model found only about 15 typo or format fixes in 480
  queries, so this data says little about how much correction helps on a real typo-heavy stream.

**What I didn't try:** a larger model, prompt variants, few-shot examples, grounding the prompt
with catalogue vocabulary, or LLM outputs as LambdaMART features. Those are the next experiments.
