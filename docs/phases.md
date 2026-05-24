# Phases

| # | Phase | Script | Key output | Smoke test | Full run |
|---|---|---|---|---|---|
| 1 | Ingest & Clean | `src/ingest.py` | `transactions` table | done | done |
| 2 | EDA | `src/eda.py` | Quality report, tier counts | done | done |
| 3 | Feature Engineering | `src/features.py` | `customers`, `products` tables | done | done |
| 4 | Holdout Construction | `src/holdout.py` | `holdout` table | done | done |
| 5 | Collaborative Filtering | `src/cf.py` | `cf_recommendations`, baseline metrics | done | done |
| 6 | LLM Runs (llm_base, llm_cf) | `src/llm.py` | `llm_results` table | todo (rerun pending) | todo |
| 7 | Evaluation | `src/evaluate.py` | `eval_results`, `run_log` rows | todo (rerun pending) | todo |

**Smoke test:** 50 customers, `dev_model_id` (Haiku), all phases end-to-end.
**Full run:** all customers, `model_id` (Sonnet). Do not start until smoke test column is all `done`.
Values from Project Config in CLAUDE.md.

---

## Phase 1 — Ingest & Clean

**Goal:** Download the raw dataset, clean it, and load it into SQLite as the foundation for all downstream work.

**Inputs:** UCI Online Retail II — direct download of `online_retail_II.xlsx` from the UCI archive (`ucimlrepo` lists id=502 but does not serve it).

**Steps:**
- [x] Download `online_retail_II.xlsx` from `https://archive.ics.uci.edu/ml/machine-learning-databases/00502/` into `data/raw/`; read both sheets with `pd.read_excel(sheet_name=None)` and concat
- [x] Rename columns to snake_case: `invoice_id`, `stock_code`, `description`, `quantity`, `invoice_date`, `price`, `customer_id`, `country`
- [x] Parse `invoice_date` to datetime
- [x] Drop rows where `customer_id` is null
- [x] Exclude invoices where `invoice_id` starts with `cancellation_prefix` (from project config)
- [x] Drop rows where `quantity <= 0` or `price <= 0`
- [x] Write to `transactions` table (CREATE TABLE IF NOT EXISTS, then DELETE + append for idempotent reload)

**Outputs:** `transactions` table

**Validation:**
- [x] Print row count before and after cleaning. Actual: 1,067,371 → 805,549 rows (the dataset covers 2009–2011 across two sheets; original ~400k → ~380k estimate referred to the 2010–2011 sheet only).
- [x] Assert no null `customer_id` remain
- [x] Assert no invoice_ids starting with cancellation prefix remain
- [x] Print date range. Actual: 2009-12-01 → 2011-12-09 (~2 years).

**Gotchas:**
- `ucimlrepo` package returns `DatasetNotFoundError` for id=502 even though the dataset is listed — fall back to the archive `.xlsx`
- The xlsx contains two sheets (`Year 2009-2010`, `Year 2010-2011`); concat both
- `customer_id` comes through as float (e.g. `12345.0`) — cast to int then string
- `(invoice_id, stock_code)` is NOT unique — same invoice can list the same product on multiple line items; do not declare it as a primary key (Phase 2 audits these duplicates)
- Requires `openpyxl` to read the xlsx (added to `requirements.txt`)

---

## Phase 2 — EDA

**Goal:** Understand data distributions and surface any quality issues before feature engineering locks in design decisions.

**Inputs:** `transactions` table

**Steps:**
- [x] Create `eda_output_dir` if it doesn't exist (path from project config)
- [x] Check shape, dtypes, null counts per column — save summary to `quality_report.txt`
- [x] Flag duplicate invoice+stock_code combinations — log count to console (actual: 36,684)
- [x] Report negative quantity and zero price counts (already excluded, but verify) — both 0 as expected
- [x] Compute purchase count per customer; assign frequency tier labels using thresholds from project config — save counts table to `freq_tier_counts.csv`
- [x] Report top-20 products by transaction volume — save to `top_products.csv`; flag products with only one transaction (actual: 122)
- [x] Plot monthly transaction volume — save to `monthly_volume.png`
- [x] Report cancellation rate by country (re-load raw data for this step only) — save to `cancellation_by_country.csv`

**Outputs:** `outputs/eda/` — see table below

| File | Contents |
|---|---|
| `quality_report.txt` | Shape, dtypes, null counts, duplicate counts |
| `freq_tier_counts.csv` | Customer count and % per frequency tier |
| `top_products.csv` | Top-20 products by transaction volume |
| `monthly_volume.png` | Monthly transaction volume line chart |
| `cancellation_by_country.csv` | Cancellation rate per country |

**Validation:**
- [x] All five output files exist after the run
- [x] Frequency tier counts must be non-zero for all tiers; log a warning if any tier has fewer than 10 customers (all tiers ≥ 140)
- [x] Cold + sparse customers should be the majority of the base (actual: 69.4%)

**Actual tier distribution (5,878 customers total):**
| Tier | Customers | % |
|---|---|---|
| cold | 2,567 | 43.67 |
| sparse | 1,510 | 25.69 |
| moderate | 1,331 | 22.64 |
| rich | 330 | 5.61 |
| champion | 140 | 2.38 |

**Gotchas:**
- EDA runs on cleaned `transactions` — cancellations are already excluded; load raw data separately for the cancellation audit step
- Use `matplotlib.use("Agg")` so the plot saves without a display
- `matplotlib` was missing from `requirements.txt`; added during Phase 2

---

## Phase 3 — Feature Engineering

**Goal:** Build customer-level RFM features and product-level metadata needed by the LLM context tiers.

**Inputs:** `transactions` table

**Customer feature steps:**
- [x] Compute recency (days from reference date = max invoice_date in dataset), frequency (distinct invoice count), monetary (sum of quantity × price) per customer
- [x] Log-transform all three (add 1 before log to handle zeros)
- [x] Standardize with `StandardScaler`
- [x] Fit `KMeans(n_clusters=kmeans_k, random_state=random_seed)` on standardized RFM
- [x] Label segments descriptively by centroid rank: champions, loyal, at_risk, hibernating (k=4)
- [x] Assign `freq_tier` from project config thresholds
- [x] Write to `customers` table (5,878 rows)

**Product feature steps:**
- [x] Count total units sold per product; rank ascending (1 = most popular)
- [x] Compute monthly sales per product; calculate CV (std/mean) as seasonality index
- [x] Flag products where CV > `seasonality_cv_threshold` (from project config) — 3,785 of 4,631 products flagged
- [x] Write to `products` table (4,631 rows)

**Outputs:** `customers` table, `products` table

**Validation:**
- [x] Segment label distribution: no single segment should exceed 60% of customers (max: hibernating 35.06%)
- [x] `freq_tier` distribution should match EDA counts from Phase 2 (matches exactly: 2567/1510/1331/330/140)
- [x] Seasonality index range should be 0–3; flag any outliers above 5 (actual range 0.000–4.314, no outliers above 5)

**Segment labeling rule:**
KMeans fits 4 clusters on standardized log(RFM). Each centroid is scored as
`score = -recency_z + frequency_z + monetary_z` (all in standardized log space),
then centroids are sorted by score descending and named in order:
`champions`, `loyal`, `at_risk`, `hibernating`. This is a one-dimensional best→worst
projection of the 3D centroid space, not a separate rule per segment. Caveat: because
recency dominates the score, `loyal` (recent, low-freq) ranks above `at_risk`
(lapsed, mid-freq) even though `at_risk` has higher frequency/monetary.

**Actual segment distribution and median RFM:**
| Segment | n | % | Recency (days) | Frequency (invoices) | Monetary (£) |
|---|---|---|---|---|---|
| champions | 1,115 | 18.97 | 15 | 13 | 5,356 |
| loyal | 1,212 | 20.62 | 22 | 3 | 744 |
| at_risk | 1,490 | 25.35 | 162 | 5 | 1,613 |
| hibernating | 2,061 | 35.06 | 399 | 1 | 295 |

**Gotchas:**
- Reference date must be the max date in the *full* dataset, not per-customer — use a single global max before any filtering
- A few customers may have monetary = 0 after exclusions; log and drop them before clustering (none in this run)
- The Phase 2 seasonality range estimate of "0–3" was conservative; the actual upper bound on this dataset is ~4.3 (still under the >5 outlier threshold)

---

## Phase 4 — Holdout Construction

**Goal:** Create per-customer ground truth by holding out each customer's most recent invoice for evaluation.

**Inputs:** `transactions` table, `customers` table (for freq_tier labels)

**Steps:**
- [x] For each customer, identify their most recent invoice by `invoice_date`
- [x] Extract all `stock_code` values from that invoice as ground truth positives
- [x] Exclude customers with only one distinct invoice (no training data would remain) — 1,623 excluded
- [x] Join `freq_tier` from `customers` table
- [x] Write to `holdout` table (4,255 customers, 83,672 ground-truth rows)
- [x] Verify all freq_tiers are represented; report counts per tier

**Outputs:** `holdout` table

**Validation:**
- [x] No customer should appear more than once in `holdout` (one held-out invoice per customer) — 4,255 distinct (customer, invoice) pairs == 4,255 customers
- [x] All five freq_tiers must have at least one customer represented; warn if any tier has fewer than 5 (all well above)
- [x] Holdout customer count should be < total customer count (some dropped due to single invoice) — 4,255 < 5,878

**Actual holdout customers per freq_tier:**
| Tier | Holdout customers | (vs. Phase 2 total) |
|---|---|---|
| cold | 944 | 2,567 — most cold customers had a single invoice |
| sparse | 1,510 | 1,510 |
| moderate | 1,331 | 1,331 |
| rich | 330 | 330 |
| champion | 140 | 140 |

**Gotchas:**
- Some invoices contain only one product — that's fine, it's still a valid ground truth
- Ties in invoice_date (same customer, same date, two invoices): pick the one with the higher invoice_id as a tiebreaker
- The `holdout` PK is `(customer_id, stock_code)` — duplicate line items on the held-out invoice (same product listed twice) are deduped before insert

---

## Phase 5 — Collaborative Filtering

**Goal:** Train an ALS collaborative filter as the non-LLM baseline and generate top-K recommendations per customer.

**Inputs:** `transactions` table (excluding holdout invoices), `holdout` table

**Steps:**
- [x] Filter `transactions` to exclude each customer's holdout invoice
- [x] Build a sparse customer × product interaction matrix (values = total quantity per customer/product, implicit feedback). Actual: 5,878 × 4,623, 426,869 nonzeros (1.57% density).
- [x] Fit `implicit.als.AlternatingLeastSquares(factors=50, iterations=20, regularization=0.01, random_state=42)` with confidence weighting `c = 1 + 40 · quantity`
- [x] For each customer, generate top-K recommendations (K=10), `filter_already_liked_items=True`
- [x] Write to `cf_recommendations` table with columns: `customer_id`, `stock_code`, `rank`, `score`
- [x] Evaluate against `holdout`: HR@K, NDCG@K per customer, coverage across all customers
- [x] Write per-customer rows to `eval_results` (model=`cf`, tier=`baseline`)
- [x] Aggregate and write summary row to `run_log`

**Outputs:** `cf_recommendations` table, rows in `eval_results` and `run_log`

**Validation:**
- [~] CF HR@K for cold customers — **expectation reversed**. Cold tier has the *highest* HR@K (0.30), not near 0. See "Already-liked masking artifact" below.
- [~] CF HR@K for champion customers — **expectation reversed**. Champions have the *lowest* HR@K (0.014).
- [x] Coverage > 1% of catalog. Actual: **57.7%** (2,671 / 4,631 products).

**Actual baseline metrics (run_id logged in `run_log`):**
| Metric | Value |
|---|---|
| HR@10 (overall) | 0.1556 |
| NDCG@10 (overall) | 0.0239 |
| Coverage | 57.68% |

**HR@10 by frequency tier:**
| Tier | HR@10 | n customers |
|---|---|---|
| cold | 0.3008 | 944 |
| sparse | 0.1748 | 1,510 |
| moderate | 0.0729 | 1,331 |
| rich | 0.0455 | 330 |
| champion | 0.0143 | 140 |

**Already-liked masking artifact (decision: keep, not fix):**
The Phase 4 holdout is each customer's most recent full invoice. Active customers (champion/rich) tend to *re-buy* products from earlier invoices, so most of their held-out products are already in their training history. With `filter_already_liked_items=True`, CF excludes those products from the candidate pool entirely — making them unrecoverable by construction. Cold/sparse customers' last invoices contain a higher share of *new* products, which CF can score normally.

We are **keeping** `filter_already_liked_items=True` for all models (CF and LLM tiers). This makes the benchmark a test of **novel-item discovery from each context tier**, not repeat-purchase prediction. The LLM tiers will face the same masking constraint, so the comparison is apples-to-apples. Phase 7 commentary should interpret tier-stratified results in this light: the question is whether richer context lets the LLM beat CF for cold customers (where novelty is high) and whether anything can lift active-customer scores given the masking floor.

**Gotchas:**
- `implicit` v0.7+ takes a user×item matrix in `model.fit(...)` (it transposes internally) — do **not** transpose manually
- Must mask already-purchased items via `filter_already_liked_items=True` (this is what creates the artifact above — see decision)
- Use `model.recommend(np.arange(n), matrix, N=K, filter_already_liked_items=True)` for batch generation — much faster than per-customer loops

---

## Phase 6 — LLM Runs (llm_base, llm_cf)

**Goal:** Generate recommendations for both LLM groups, checkpointed for resumability.

**Inputs:** `transactions`, `products`, `cf_recommendations` tables

**For each group llm_base → llm_cf:**
- [x] Load customers to process; filter out those already in `llm_results` for this group + run_id (checkpoint)
- [x] For each customer, assemble the group-appropriate context block (see `skills/llm_prompts.md`)
- [x] Call the API using `model_id` (full run) or `dev_model_id` (smoke test)
- [x] Parse JSON response; on failure log the raw response and insert a null row — do not abort
- [x] Write result to `llm_results` immediately after each call
- [x] Log progress to console every `checkpoint_every` calls

**Outputs:** `llm_results` table (one row per customer per group)

**Validation (smoke, n=100, Haiku):**
- [ ] After each group completes, verify row count equals expected customer count
- [ ] Parse failure rate should be < 5%; investigate prompt if higher
- [ ] Spot-check 3–5 rationale strings to confirm the model is using the context (rationales reference history; `llm_cf` rationales reference CF neighbors)

**Design note (T1/T3 dropped, renamed — 2026-04-27 / 2026-05-23):** earlier iterations had four tiers T0–T3. T1 (segment + global popularity) was redundant with the popularity-curated candidate list and empirically hurt HR@10; T3's churn/propensity labels were deterministic functions of recency (an input feature) — target leakage. The remaining T0/T2 were renamed `llm_base` and `llm_cf` to lead with the three-group framing: **`cf_baseline`** (ML-only) vs **`llm_base`** (LLM-only) vs **`llm_cf`** (LLM+CF).

**Gotchas:**
- Run groups in `llm_base` → `llm_cf` order; `llm_cf`'s prompt builds on the `llm_base` context block
- Smoke test uses `dev_model_id` (Haiku) and `sample_size` customers — verify both are set before starting
- Anthropic tier-1 rate limits: 50 RPM and 10K output tokens/min on Haiku — keep `--workers 4 --rpm 45` for the smoke; full Sonnet run will need a similar rate limit
- The model returns recommendations as `"description (STOCKCODE)"` when given a `- description (CODE)` candidate list — prompt explicitly demands bare codes, and `parse_response` has a `(CODE)` regex fallback as belt-and-suspenders
- Strict "must pick from candidate list" prompts cause empties to balloon for cold customers; the candidate list is positioned as *suggestions* and validation only enforces `catalog - purchased`

---

## Phase 7 — Evaluation

**Goal:** Compute HR@K, NDCG@K, and coverage for all models and tiers; stratify by frequency tier and RFM segment.

**Inputs:** `holdout`, `cf_recommendations`, `llm_results`, `customers` tables

**Steps:**
- [x] For each (model, tier) pair — cf_baseline + llm_base + llm_cf:
  - [x] Join recommendations with holdout per customer
  - [x] Compute HR@K: 1 if any holdout product appears in top-K, else 0
  - [x] Compute NDCG@K: position-discounted relevance score
  - [x] Write one row per customer to `eval_results`
- [x] Compute coverage: unique products recommended / total products in catalog
- [x] Aggregate means by freq_tier and by RFM segment
- [x] Write one summary row per (model, tier) to `run_log`

**Outputs:** `eval_results` table, `run_log` summary rows

**Validation (smoke, n=100 — rerun pending after T1/T3 removal):**
- [ ] `llm_cf` should beat `llm_base` (CF context helps the LLM)
- [ ] `llm_cf` should match or beat `cf_baseline` (LLM re-ranks CF neighbors usefully)
- [ ] LLM groups should outperform `cf_baseline` for **cold** tier (novelty is high there)
- [ ] For **rich/champion** tiers, all models heavily depressed by masking artifact (Phase 5 note)

**Gotchas:**
- LLM recommendations are returned as product descriptions, not stock codes — must fuzzy-match or normalize to stock codes before comparing with holdout
- Customers with no recommendations (parse failure rows) should be excluded from metric aggregation and noted in run_log
- Apply the same `filter_already_liked_items` masking to LLM recommendations as CF — exclude any recommended product the customer already bought before the held-out invoice — so CF and LLM are evaluated on identical candidate sets (see Phase 5 decision)
