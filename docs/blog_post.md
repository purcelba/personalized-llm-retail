---
layout: post
title: Personalized Retail Recommendations | Can LLMs Beat Collaborative Filtering?
---
<br>

Product recommendations are one of the longest-running applications of machine learning in industry. Collaborative filtering (CF) — "users who look like you preferred these products" — has been the workhorse for two decades. Modern large language models bring something CF doesn't: the ability to reason over the full text of a customer's purchase history and a catalog of product descriptions, and to recommend items with semantic associations the matrix can't see.

The obvious question is whether the LLM is a *replacement* for CF, an *enhancement* on top of it, or neither. Published results exist - [WeChat's LEADRE](https://arxiv.org/abs/2411.13789) reports +1.57% GMV from an LLM-augmented ad recommender in production, and [academic e-commerce benchmarks](https://arxiv.org/abs/2410.12829) report precision and recall gains in the 7–10pp range - but they're typically compared against unspecified production baselines. Practitioner-oriented apples-to-apples comparisons of LLM vs. CF on a single business's data, with paired statistical testing, are harder to find. I wanted one.

To get one, I built a three-way benchmark on a real transactional dataset, evaluated paired against the same held-out invoices, and computed bootstrap confidence intervals on the differences. The project had three goals:

- **I designed a controlled benchmark** comparing a classical CF baseline against two LLM configurations on identical customers and ground truth, with paired statistical comparisons.
- **I tested the hypothesis** that an LLM combined with CF signal would outperform either alone — and quantified by how much.
- **I translated the results into recommendations** for teams considering LLM-based personalization in production.

# Key Takeaways

The benchmark suggests three things a practitioner can take to the bank.

- **An LLM alone does not beat CF.** Sonnet 4.6 reasoning over purchase history and a popularity-curated shortlist is statistically indistinguishable from ALS collaborative filtering on overall accuracy. The "LLMs will eat ML" framing isn't supported here.
- **An LLM *plus* CF beats both.** Feeding the LLM the CF model's top neighbors as context flips the result: HR@10 +2.1pp (p≈0.06), NDCG@10 +25% (p=0.008) over CF alone. The win is composition, not substitution.
- **The advantage shows up where it matters most — mid-frequency customers.** For customers with 3–5 prior invoices, the LLM+CF stack beats CF by **+6.8pp HR@10 (p<0.01)**. That's the cohort where retailers actually have leverage; newer customers who haven't fully revealed themselves to a collaborative filter.

| Customer group | CF baseline | LLM + CF | Δ HR@10 |
|---|---:|---:|---:|
| Cold (1–2 invoices) | 29.2% | 25.1% | −4.2pp |
| Sparse (3–5 invoices) | 13.6% | **20.4%** | **+6.8pp** |
| Moderate (6–15 invoices) | 8.4% | 11.2% | +2.8pp |
| Rich (16–30 invoices) | 4.5% | 7.0% | +2.4pp |
| Champion (31+ invoices) | 1.4% | 3.6% | +2.2pp |

The full code, data, and analysis are in the [project repository](https://github.com/purcelba/personalized-llm-retail).

# Data Overview

The benchmark uses the **UCI Online Retail II** dataset — a real transactional ledger from a UK-based online gift retailer covering 2009–2011. After cleaning (dropping cancellations, null customers, and quantity ≤ 0 rows), the dataset contains 805,549 line items across **5,878 customers** and **4,631 unique products**.

The customer base has a long-tail distribution typical of real retail: 69% of customers have ≤ 5 invoices, only 2.4% have more than 30. I assigned each customer to a frequency tier (`cold` / `sparse` / `moderate` / `rich` / `champion`) and stratified all evaluations by tier, since the underlying difficulty varies sharply across them.

For ground truth, I held out each customer's most recent invoice and used it as a multi-positive target. This gives 4,255 evaluable customers (those with at least two invoices in the dataset) and ~84,000 ground-truth product-customer pairs to score against.

# Methods

I compared three recommenders. All three are scored on the same task: recommend 10 products the customer has not already purchased, score against the held-out invoice using HR@10 and NDCG@10.

**`cf_baseline`** — Alternating Least Squares (ALS) collaborative filtering from the `implicit` library (`factors=50, iterations=20, alpha=40`). The customer × product purchase matrix is factorized into two dense matrices; recommendations are dot-product scores over the product factor matrix, with already-purchased items masked out. ALS is the industry-default baseline for implicit-feedback recommendation on transactional data.

**`llm_base`** — Claude Sonnet 4.6 given a prompt containing the customer's top-25 purchase history (description + units) and a 50-item candidate list drawn from the top-200 most popular products in the catalog, with already-purchased items excluded. The model returns 10 stock codes plus a free-text rationale.

**`llm_cf`** — Same as `llm_base`, plus one extra block in the prompt: the top-10 CF neighbors for that customer, also added to the candidate pool. This is the only difference between the two LLM configurations, so any delta is attributable to the CF-neighbor signal.

The two LLM configurations share a candidate-list scaffold — the model picks from ~50 codes, not the full 4,631-product catalog — because earlier iterations that let the model pick freely collapsed into ~50% invalid output (hallucinated stock codes, echoes of purchase history). The candidate list is a guardrail against hallucination, not a neutral baseline. I return to this limitation below.

For evaluation I used **HR@10** (binary: did at least one held-out product appear in the top-10?) and **NDCG@10** (position-discounted relevance). All comparisons are paired on the customer intersection where both LLM groups returned parseable output (n=1,612 customers). I computed 95% confidence intervals on each metric using 10,000 bootstrap resamples, and paired bootstrap CIs on the differences.

# Result 1 — An LLM Alone Doesn't Beat CF, But an LLM + CF Does

Across the n=1,612 paired customers, the headline numbers are:

| Group | HR@10 | NDCG@10 | Coverage |
|---|---|---|---:|
| cf_baseline | 0.132 [0.116, 0.148] | 0.0196 [0.017, 0.023] | 46% |
| llm_base | 0.137 [0.120, 0.155] | 0.0216 [0.018, 0.025] | 3% |
| llm_cf | **0.151 [0.133, 0.169]** | **0.0245 [0.021, 0.028]** | 17% |

The paired comparisons:

| Δ | HR@10 (95% CI) | p | NDCG@10 (95% CI) | p |
|---|---|---|---|---|
| llm_cf − cf_baseline | +0.019 [+0.000, +0.038] | 0.056 | +0.005 [+0.001, +0.009] | **0.008** |
| llm_cf − llm_base | +0.014 [+0.001, +0.026] | **0.034** | +0.003 [+0.001, +0.005] | **0.016** |
| llm_base − cf_baseline | +0.006 [−0.016, +0.026] | 0.64 | +0.002 [−0.002, +0.006] | 0.34 |

![Overall HR and NDCG]({{ site.baseurl }}/images/personalized-retail/fig1_overall.png "Overall HR and NDCG."){: .center-image }
<p align="center">
<font size="2"><b>Figure 1.</b> Overall HR@10 and NDCG@10 across the three recommenders (n=1,612 paired customers). Error bars are 95% bootstrap CIs. LLM + CF leads both metrics; LLM alone is indistinguishable from CF.</font>
</p>

Two things stand out. First, the LLM-only configuration (`llm_base`) is statistically indistinguishable from CF on both metrics — confidence intervals on the differences sit squarely on zero. The LLM reasoning over purchase history plus popularity is, on average, no better and no worse than 20 years of matrix factorization. Second, adding CF neighbors to the prompt produces a real and significant lift over both CF and `llm_base`. The NDCG advantage is well outside the noise floor.

The interpretation is mechanistic. The LLM isn't replacing the recommender — it's reading CF's output and re-ranking it using product-description semantics ALS doesn't have access to. CF surfaces ten plausibly relevant products; the LLM looks at the customer's history, looks at the candidate descriptions, and picks the subset most likely to land. That's a different skill than ALS's job of generating candidates in the first place.

# Result 2 — The Win Concentrates in Mid-Frequency Customers

The overall numbers undersell the size of the effect for the customer cohort where personalization actually pays off.

![HR by frequency tier]({{ site.baseurl }}/images/personalized-retail/fig2_by_tier.png "HR by frequency tier."){: .center-image }
<p align="center">
<font size="2"><b>Figure 2.</b> HR@10 by frequency tier for the three recommenders. Bars are 95% bootstrap CIs. The sparse tier (3–5 invoices) is where the LLM + CF lift is largest and statistically clearest.</font>
</p>

The clearest win is the **sparse** tier — customers with 3 to 5 prior invoices. There, `llm_cf` beats CF by +6.8pp HR@10 with a 95% CI of [+2.4, +11.4] and p<0.01. NDCG is +1.8pp, also significant. These are customers with enough history for an LLM to reason about taste (favored categories, gift vs. self-purchase patterns) but not so much that ALS's collaborative signal has fully resolved them. That's precisely the regime where adding text-based reasoning over product descriptions adds something the matrix can't see.

Moderate and rich tiers show smaller positive gains in the same direction. The champion tier (31+ invoices) is too small in this dataset (n=139) for the comparison to clear significance on its own, though the direction is consistent.

# Result 3 — The Lift Has a Price Tag

The accuracy comparison is only half the story. The three approaches have fundamentally different cost structures, and the right model depends on traffic and unit economics, not just HR@10.

Cost per 1,000 customer-recommendation requests, using Claude Sonnet 4.6 pricing ($3/M input tokens, $15/M output tokens) and the actual prompt sizes from this benchmark:

| Approach | Per call | Per 1,000 customers | Notes |
|---|---:|---:|---|
| `cf_baseline` | ~$0 | **~$0** | One ALS fit (~10s on this dataset). Each recommendation is a dot product. Effectively free at recommendation time. |
| `llm_base` | ~$0.0086 | **~$8.60** | ~850 input + ~400 output tokens per call. |
| `llm_cf` | ~$0.0089 | **~$8.90** | ~970 input + ~400 output tokens. Requires CF infrastructure underneath. |

![Cost per 1,000 customers]({{ site.baseurl }}/images/personalized-retail/fig3_cost.png "Cost per 1,000 customers."){: .center-image }
<p align="center">
<font size="2"><b>Figure 3.</b> Estimated cost per 1,000 customer-recommendation requests. CF is effectively free at recommendation time; LLM approaches add ~$9 per 1,000 customers at Sonnet 4.6 pricing.</font>
</p>

A few practical implications.

- **Cost shape, not just magnitude.** CF is fixed training cost plus near-zero per-customer serving. LLM is pay-per-recommendation, scaling linearly with traffic — ~$89K/year for 100K customers refreshed weekly.
- **Pre-compute, don't serve live.** Run `llm_cf` once per customer per week and cache. That's the design assumed in the numbers above.
- **Route by segment.** The clean lift is in the sparse tier — send only that cohort through `llm_cf` and serve CF for the rest. Cuts cost-per-customer by an order of magnitude while preserving the lift where it matters.
- **Haiku for dev, Sonnet for prod.** Haiku 4.5 is ~5× cheaper but had higher parse failures and weaker NDCG in testing.

**The bottom line on cost.** Use the offline benchmark to decide *what to A/B test*, not whether to ship — offline HR@10 lift typically compresses several-fold when measured online, and only an A/B test on incremental gross profit (revenue lift × margin minus LLM spend) tells you whether `llm_cf` actually pays back. The recommended starting experiment is a holdout test on the sparse tier alone, where the offline signal is strongest and the cost exposure is smallest.

# Limitations

A few caveats are worth surfacing directly.

**The `llm_base` comparison isn't unaided.** I feed the LLM a 50-item popularity-filtered candidate list, so it never gets a fair shot at the long-tail ~4,400 products. ALS, by contrast, scores the entire 4,631-product catalog for every customer. The cleanest version of the experiment — giving the LLM a product-search tool and letting it query the full catalog — would require a substantially more expensive setup and was out of scope here. The takeaway is narrower than "LLMs can't beat CF." It's "an LLM reasoning over a popularity shortlist can't beat CF on its own, but it can re-rank CF's output to beat both."

**Coverage is wildly different across groups.** CF uses 46% of the catalog; llm_cf uses 17%; llm_base uses 3%. Low coverage means the LLM-only configuration is essentially a popularity ranker dressed in chain-of-thought. The CF-neighbor block restores enough breadth to look like real personalization. This is part of why llm_cf wins NDCG even when HR is close.

**Cost is non-trivial.** The Sonnet run cost about $25 for n≈1,600 customers across both LLM groups; a full-catalog production deployment would run into real money. The relevant question for any practitioner is whether a few percentage points of HR on mid-frequency customers justifies the API spend versus a free ALS retrain.

# Conclusions

Three practical implications fall out of this.

- **LLMs and CF are complementary, not competing.** The biggest, most reliable win comes from giving the LLM the CF model's output as context — using the LLM as a semantic re-ranker on top of a collaborative filter. Teams considering "replace our recommender with an LLM" are choosing the wrong frame.
- **The effect is sharpest in mid-frequency customers**, who are also the most commercially interesting cohort. Cold-start has well-developed solutions (popularity + onboarding); whales recommend themselves. The sparse/moderate band is where personalization is hard and where this stack actually moves the needle.
- **The benchmark itself matters more than vendors usually admit.** `filter_already_liked_items=True` is a near-universal production default that makes high-frequency customers structurally unrecoverable on novel-item HR. Any honest comparison has to stratify by frequency and report paired CIs; aggregate numbers can flip the headline in either direction.

The next natural extension is replacing the candidate-list scaffold with tool-use — giving the LLM a product-search function and letting it pull from the full catalog. That would test the actual ceiling on what an unaided LLM can do, and would let the comparison run apples-to-apples on candidate-space size. The full code, prompts, evaluation harness, and bootstrap analysis are available on [GitHub](https://github.com/purcelba/personalized-llm-retail).
