# Retail Personalization — LLM Recommendation Benchmark

## Goal
Build a personalized product recommendation system using the UCI Online Retail II dataset.
Benchmark LLM recommendation quality across two context enrichment tiers against a
collaborative filtering baseline. Store all outputs in a SQLite database queryable
via natural language.

---

## Dataset

**UCI Online Retail II** (ucimlrepo id=502, originally curated by Dr. Daqing Chen).

A real **transactional ledger** from a UK-based online gift retailer covering **2009-12-01 → 2011-12-09** (~2 years across two xlsx sheets: `Year 2009-2010` and `Year 2010-2011`). Each row is one line item on one invoice — i.e. a single product, quantity, and price at the moment of purchase. The company specializes in unique all-occasion giftware; many customers are wholesalers, which is why per-invoice quantities can be large.

### Raw fields
| Column (cleaned) | Type | Description |
|---|---|---|
| `invoice_id` | string | 6-digit invoice number; a single value groups all line items from one transaction. Invoices prefixed `C` are **cancellations** (always excluded — see "key quirk" below). |
| `stock_code` | string | 5-character product identifier, sometimes with a letter suffix (e.g. `85123A`). The product's stable key — descriptions can vary by line but `stock_code` is canonical. |
| `description` | string | Free-text product name (e.g. `WHITE HANGING HEART T-LIGHT HOLDER`). Same `stock_code` can carry slightly different descriptions across rows. |
| `quantity` | int | Units purchased on this line item. Negatives in raw data indicate returns — dropped in Phase 1 (`quantity <= 0` excluded). |
| `invoice_date` | datetime | Timestamp of the transaction (date + time, minute precision). |
| `price` | float | Unit price in **GBP**. Zero/negative prices dropped in Phase 1 (data quality issues). |
| `customer_id` | string | 5-digit customer identifier. Rows with null `customer_id` are dropped (anonymous transactions can't be personalized). |
| `country` | string | Customer's country of residence. Heavy UK skew (~91%); cancellation rates vary by country. |

### Scale (after Phase 1 cleaning)
| Metric | Value |
|---|---|
| Rows (transactions) | 805,549 |
| Unique customers | 5,878 |
| Unique products (stock_codes) | 4,631 |
| Date range | 2009-12-01 → 2011-12-09 |
| Total revenue | ~£17.7M |
| Median basket size | ~12 line items |
| Cancellation rate (raw, pre-clean) | ~2% of invoices |

### Why this dataset for the benchmark
- **Implicit feedback only** — purchases, not ratings. Realistic for any non-media commerce setting.
- **Long-tail customer distribution** — 69% of customers are `cold` or `sparse` (≤5 invoices), 2.4% are `champion` (>30 invoices). This is the dominant shape in real retail and is where personalization is hardest.
- **Rich textual product descriptions** — every stock_code has a human-readable name, which is what gives the LLM something to reason about beyond IDs.
- **Real-world quality issues** — duplicate line items per invoice, free-text description drift, null customer_ids, cancellations — exercises the full data-cleaning path.

### Key quirks
- **Cancellations** — `invoice_id` starting with `C` (~2% of rows). Excluded from the model pipeline; audited separately in Phase 2.
- **Duplicate line items** — the same `(invoice_id, stock_code)` pair can appear multiple times on one invoice (36,684 such duplicates). Phase 2 surfaces this; do **not** declare it a primary key.
- **`customer_id` as float** — comes through Excel as `12345.0`. Cast to int → string in Phase 1.
- **Two sheets** — the xlsx contains 2009-2010 and 2010-2011 separately; concat both. (Many published examples use only the second sheet — be careful with comparisons.)
- **`ucimlrepo` doesn't serve id=502** despite listing it. Phase 1 falls back to a direct xlsx download from the UCI archive.

---

## Database
All data, features, model outputs, and evaluation results land in `data/retail.db` (SQLite).

Core tables:
- `transactions`       — cleaned raw data
- `customers`          — RFM features + segment labels
- `products`           — popularity rank, seasonality index
- `holdout`            — ground truth last-purchase per customer
- `cf_recommendations` — collaborative filter top-K per customer
- `llm_results`        — LLM recommendations + rationale per customer per tier
- `eval_results`       — HR@K, NDCG@K per customer per model per tier
- `run_log`            — every model run with params and aggregate metrics

---

## Benchmark Design

### Frequency tiers
Thresholds defined in Project Config. Stratify all evaluations by these tiers.

### Three comparison groups
The benchmark compares three recommenders on the same customers and the same held-out invoices:

| Group | What it is | Inputs |
|---|---|---|
| **`cf_baseline`** | ML-only control (no LLM) — ALS collaborative filter | customer × product matrix |
| **`llm_base`** | LLM-only — recommends from purchase history + popularity-curated candidate list | purchase history |
| **`llm_cf`** | LLM + CF — same as `llm_base` plus a block of CF neighbor products | purchase history + CF neighbors |

`llm_cf` is strictly additive on top of `llm_base`: the only difference between the two prompts is the CF-neighbor block, so any `llm_base` → `llm_cf` delta is attributable to that signal. The three-way design isolates two questions:

- **`cf_baseline` vs `llm_base`**: can an LLM reasoning over purchase history match an ML model trained on the full customer × product matrix?
- **`llm_base` vs `llm_cf`**: does giving the LLM the CF model's output as context let it *beat* CF — i.e. re-rank or filter CF neighbors using purchase-history semantics the matrix factorization can't see?

#### What's actually in each LLM prompt

Both LLM groups share a common scaffold: a system prompt with output-format rules, a `Customer purchase history` block (top-25 prior items by units purchased), and a `CANDIDATE LIST` of 50 stock_codes to pick from. `llm_cf` adds one intermediate context block between history and candidates.

| Group | Adds these blocks | Adds to candidate pool |
|---|---|---|
| **`llm_base`** | (nothing — just history + candidates) | top-200 most popular products |
| **`llm_cf`** | `Customers with similar purchase patterns also bought: …` (top-10 CF neighbors from `cf_recommendations`) | + the same 10 CF neighbors |

#### Design caveat — `llm_base` is not "fully unaided"

Both LLM groups receive a 50-item `CANDIDATE LIST` so the model isn't asked to hallucinate stock_codes from the 4,631-product catalog. The base pool for that list is the **top 200 most popular products** in the catalog, minus anything the customer already purchased, sliced to 50.

This means popularity is already implicitly informing `llm_base` — the candidate list itself is popularity-curated. Consequences:

- **`llm_base` measures**: "what can the LLM do with purchase history + a popularity-filtered shortlist?" — not "what can the LLM do unaided?"
- **The alternative** — letting the LLM pick freely from all 4,631 codes — was tried in early Phase 6 iterations and collapsed into ~50% empty validation rates (the model invented codes or echoed purchase history). The candidate list is a pragmatic guardrail, not a neutral baseline. Future work could remove this constraint with a Sonnet-grade model + tool-use product lookup.

Source of truth for the exact prompt assembly is `src/llm.py` (`build_prompt`, `build_candidate_pool`).

#### Example: one customer end-to-end

Customer **12421** (a real record from the dataset) — `freq_tier=moderate`, `segment=loyal`, RFM `(recency=15 days, frequency=6 invoices, monetary=£1,098)`. This is the data the pipeline has on them, and how it lands in each tier's prompt.

**Source 1 — Purchase history** (from `transactions`, holdout invoice excluded) — used by *all tiers*:
| stock_code | description | units | spend |
|---|---|---|---|
| 22066 | LOVE HEART TRINKET POT | 60 | £36.12 |
| 21519 | GIN & TONIC DIET GREETING CARD | 60 | £25.20 |
| 21232 | STRAWBERRY CERAMIC TRINKET BOX | 60 | £75.00 |
| 22646 | CERAMIC STRAWBERRY CAKE MONEY BANK | 36 | £52.20 |
| 84991 | 60 TEATIME FAIRY CAKE CASES | 24 | £13.20 |
| 22893 | MINI CAKE STAND T-LIGHT HOLDER | 24 | £10.08 |
| ... | (top-25 by units shown to model) | | |

**Source 2 — Catalog signals** (from `products`, computed Phase 3) — `popularity_rank` is used to assemble the top-200 popularity pool that backs the candidate list for *both LLM groups*.

**Source 3 — CF neighbors** (from `cf_recommendations`, computed Phase 5) — added to *`llm_cf`*:
| rank | stock_code | description | ALS score |
|---|---|---|---|
| 1 | 71477 | COLOUR GLASS. STAR T-LIGHT HOLDER | 1.312 |
| 2 | 37503 | TEA TIME CAKE STAND IN GIFT BOX | 1.296 |
| 3 | 22950 | 36 DOILIES VINTAGE CHRISTMAS | 1.207 |
| 4 | 84510B | SET OF 4 FAIRY CAKES COASTERS | 1.161 |
| 5 | 35832 | WOOLLY HAT SOCK GLOVE ADVENT STRING | 1.125 |

**Ground truth — held-out invoice** (from `holdout`, computed Phase 4) — never shown to the model, only used for scoring:
`[21108, 21907, 22307, 23234, 23267, 23311, 23382, 47567B]`

**What the model sees**: a single prompt assembled from the relevant subset of the above (per group), plus a `CANDIDATE LIST` of 50 stock_codes drawn from popularity (+ CF for `llm_cf`), excluding anything the customer already bought. It returns 10 stock_codes + a rationale; we score against the ground-truth invoice using HR@10 and NDCG@10.

**`llm_cf` adds ~470 chars over `llm_base`** (3,404 → ~3,870 chars for this customer): the CF-neighbors block plus a small candidate-list shuffle. Full verbatim prompts for both groups are in [`docs/example_prompts.md`](docs/example_prompts.md).

### `cf_baseline` (non-LLM control)

The benchmark compares both LLM groups against a **collaborative filtering** (CF) baseline built in Phase 5. This is the reference the LLM groups must beat to justify their cost.

- **Algorithm**: implicit-feedback **Alternating Least Squares (ALS)** from the `implicit` library (`factors=50, iterations=20, regularization=0.01, alpha=40`).
- **Input**: a sparse customer × product matrix of purchase quantities, with the held-out invoice for each customer excluded so the baseline trains on the same signal the LLM tiers see.
- **Output**: top-10 recommendations per customer, written to `cf_recommendations`. The model factorizes the matrix into a customer-factor and product-factor matrix, then scores each candidate by their dot product; recs are the highest-scoring novel products per customer.
- **Masking**: uses `filter_already_liked_items=True` — already-purchased products are excluded from the candidate pool. The same constraint is applied to LLM recs (via `valid_codes = all_codes − purchased`), so the comparison is apples-to-apples on **novel-item discovery**, not repeat-buys. See Phase 5 in `docs/phases.md` for the resulting masking artifact (cold customers score *higher* than champions because their holdouts contain more novel items).
- **Why this baseline?** ALS is the de facto industry default for implicit-feedback recommendations on transactional data. If the LLM groups can't beat a well-tuned ALS on this dataset, the personalization story doesn't hold.

### Metrics

All metrics evaluated at K=10 against the held-out invoice per customer (see Phase 4).

- **Hit Rate @10 (HR@10)** — binary per customer: 1 if **any** held-out product appears in the top-10 recommendations, else 0. Aggregated as a mean across customers, so the reported value is the *fraction of customers* for whom we got at least one hit. Range 0–1, higher is better. Insensitive to *where* in the top-10 the hit lands.
- **NDCG @10** — normalized discounted cumulative gain. Rewards hits at higher ranks: a hit at rank 1 contributes `1/log2(2)`, rank 2 contributes `1/log2(3)`, etc. Normalized by the ideal DCG given how many held-out items the customer has (capped at K). Range 0–1, higher is better. Use HR@10 to ask *did we hit at all*, NDCG@10 to ask *how well-ranked were the hits*.
- **Coverage** — fraction of the product catalog that appears at least once across all customers' top-10 lists. Range 0–1, higher = broader catalog use. Low coverage with high HR means the recommender is concentrating on a few popular items that happen to work; high coverage with low HR means it's spreading recs too thin. A healthy recommender wants both.
- **Stratification** — report all metrics by `freq_tier` (cold/sparse/moderate/rich/champion) and `segment` (champions/loyal/at_risk/hibernating). Different tiers face different difficulty (see Phase 5 "already-liked masking artifact" note), so overall means can mislead.

---

## Skills
Read the relevant skill before starting each phase. Skills live in `skills/`.

| Skill file              | Use when...                                      |
|-------------------------|--------------------------------------------------|
| `skills/eda.md`         | Profiling data, surfacing quality issues         |
| `skills/features.md`    | Building RFM, co-occurrence, seasonality features|
| `skills/database.md`    | Schema design, loading data, writing queries     |
| `skills/evaluation.md`  | Holdout construction, metric computation         |
| `skills/llm_prompts.md` | Assembling and calling the LLM per tier          |

---

## Conventions
- Python 3.10+, pandas, scikit-learn, implicit, anthropic, sqlite3
- API key: `os.environ["ANTHROPIC_API_KEY"]` — never hardcode
- Save checkpoints every N LLM calls (N from Project Config) — runs must be resumable
- Log every model run to `run_log` table with timestamp + params + metrics
- Skill files in `skills/` contain methodology only — all project-specific values live in Project Config below

---

## Project Config
All project-specific values are centralized here. Skill files reference these by name.

| Key | Value |
|---|---|
| dataset_id | 502 (ucimlrepo) |
| db_path | `data/retail.db` |
| eda_output_dir | `outputs/eda/` |
| cancellation_prefix | `C` |
| model_id | `claude-sonnet-4-6` |
| dev_model_id | `claude-haiku-4-5-20251001` |
| sample_size | 50 customers (smoke test) / None (full run) |
| temperature | 0 |
| max_tokens | 600 |
| random_seed | 42 |
| top_k | 10 |
| kmeans_k | 4 |
| checkpoint_every | 50 |
| seasonality_cv_threshold | 0.5 |

### Frequency tier thresholds
| Tier | Min purchases | Max purchases |
|---|---|---|
| cold | 1 | 2 |
| sparse | 3 | 5 |
| moderate | 6 | 15 |
| rich | 16 | 30 |
| champion | 31 | — |

---

## Development Workflow

**Always run a smoke test before a full run.**

1. **Smoke test first** — run the full Phase 1–10 pipeline on a small sample using the dev model (see Project Config). The goal is to verify the pipeline is wired correctly end-to-end, not to produce meaningful metrics.
2. **Sign off before scaling** — only proceed to a full run after the smoke test completes without errors and eval_results rows are populated with plausible values.
3. **Full run** — rerun all phases with `sample_size=None` and `model_id` (Sonnet).

When starting any phase, check `docs/phases.md` status. If the smoke test column is not marked `done`, implement and run the smoke test before the full run.

---

## Execution order
See `docs/phases.md` for status and detail on each phase.
