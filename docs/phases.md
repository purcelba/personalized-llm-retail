# Phases

| # | Phase | Script | Key output | Smoke test | Full run |
|---|---|---|---|---|---|
| 1 | Ingest & Clean | `src/ingest.py` | `transactions` table | todo | todo |
| 2 | EDA | `src/eda.py` | Quality report, tier counts | todo | todo |
| 3 | Feature Engineering | `src/features.py` | `customers`, `products` tables | todo | todo |
| 4 | Holdout Construction | `src/holdout.py` | `holdout` table | todo | todo |
| 5 | Collaborative Filtering | `src/cf.py` | `cf_recommendations`, baseline metrics | todo | todo |
| 6 | LLM Runs (T0–T3) | `src/llm.py` | `llm_results` table | todo | todo |
| 7 | Evaluation | `src/evaluate.py` | `eval_results`, `run_log` rows | todo | todo |
| 8 | Reporting | `src/report.py` | Console summary + plots | todo | todo |
| 9 | NL Query Interface | `src/nl_query.py` | Interactive query loop | todo | todo |
| 10 | Recommendations API | `src/api.py` | FastAPI server serving pre-computed recs | todo | todo |

**Smoke test:** 50 customers, `dev_model_id` (Haiku), all phases end-to-end.
**Full run:** all customers, `model_id` (Sonnet). Do not start until smoke test column is all `done`.
Values from Project Config in CLAUDE.md.

---

## Phase 1 — Ingest & Clean

**Goal:** Download the raw dataset, clean it, and load it into SQLite as the foundation for all downstream work.

**Inputs:** UCI Online Retail II via `ucimlrepo` (dataset_id from project config)

**Steps:**
- [ ] `fetch_ucirepo(id=dataset_id)` — returns a dict with `data.features` and `data.targets`; combine into one DataFrame
- [ ] Rename columns to snake_case: `invoice_id`, `stock_code`, `description`, `quantity`, `invoice_date`, `price`, `customer_id`, `country`
- [ ] Parse `invoice_date` to datetime
- [ ] Drop rows where `customer_id` is null
- [ ] Exclude invoices where `invoice_id` starts with `cancellation_prefix` (from project config)
- [ ] Drop rows where `quantity <= 0` or `price <= 0`
- [ ] Write to `transactions` table (CREATE TABLE IF NOT EXISTS + INSERT OR REPLACE)

**Outputs:** `transactions` table

**Validation:**
- [ ] Print row count before and after cleaning; expect ~400k → ~380k rows
- [ ] Assert no null `customer_id` remain
- [ ] Assert no invoice_ids starting with cancellation prefix remain
- [ ] Print date range to confirm dataset spans ~2 years

**Gotchas:**
- `ucimlrepo` returns the two annual sheets already combined — no manual merge needed
- `customer_id` comes through as float (e.g. `12345.0`) — cast to int then string

---

## Phase 2 — EDA

**Goal:** Understand data distributions and surface any quality issues before feature engineering locks in design decisions.

**Inputs:** `transactions` table

**Steps:**
- [ ] Create `eda_output_dir` if it doesn't exist (path from project config)
- [ ] Check shape, dtypes, null counts per column — save summary to `quality_report.txt`
- [ ] Flag duplicate invoice+stock_code combinations — log count to console
- [ ] Report negative quantity and zero price counts (already excluded, but verify)
- [ ] Compute purchase count per customer; assign frequency tier labels using thresholds from project config — save counts table to `freq_tier_counts.csv`
- [ ] Report top-20 products by transaction volume — save to `top_products.csv`; flag products with only one transaction
- [ ] Plot monthly transaction volume — save to `monthly_volume.png`
- [ ] Report cancellation rate by country (re-load raw data for this step only) — save to `cancellation_by_country.csv`

**Outputs:** `outputs/eda/` — see table below

| File | Contents |
|---|---|
| `quality_report.txt` | Shape, dtypes, null counts, duplicate counts |
| `freq_tier_counts.csv` | Customer count and % per frequency tier |
| `top_products.csv` | Top-20 products by transaction volume |
| `monthly_volume.png` | Monthly transaction volume line chart |
| `cancellation_by_country.csv` | Cancellation rate per country |

**Validation:**
- [ ] All five output files exist after the run
- [ ] Frequency tier counts must be non-zero for all tiers; log a warning if any tier has fewer than 10 customers
- [ ] Cold + sparse customers should be the majority of the base

**Gotchas:**
- EDA runs on cleaned `transactions` — cancellations are already excluded; load raw data separately for the cancellation audit step
- Use `matplotlib.use("Agg")` so the plot saves without a display

---

## Phase 3 — Feature Engineering

**Goal:** Build customer-level RFM features and product-level metadata needed by the LLM context tiers.

**Inputs:** `transactions` table

**Customer feature steps:**
- [ ] Compute recency (days from reference date = max invoice_date in dataset), frequency (distinct invoice count), monetary (sum of quantity × price) per customer
- [ ] Log-transform all three (add 1 before log to handle zeros)
- [ ] Standardize with `StandardScaler`
- [ ] Fit `KMeans(n_clusters=kmeans_k, random_state=random_seed)` on standardized RFM
- [ ] Label segments descriptively by centroid rank (e.g. champions = high F+M, low R; at_risk = high R, declining F+M)
- [ ] Assign `freq_tier` from project config thresholds
- [ ] Write to `customers` table

**Product feature steps:**
- [ ] Count total units sold per product; rank ascending (1 = most popular)
- [ ] Compute monthly sales per product; calculate CV (std/mean) as seasonality index
- [ ] Flag products where CV > `seasonality_cv_threshold` (from project config)
- [ ] Write to `products` table

**Outputs:** `customers` table, `products` table

**Validation:**
- [ ] Segment label distribution: no single segment should exceed 60% of customers
- [ ] `freq_tier` distribution should match EDA counts from Phase 2
- [ ] Seasonality index range should be 0–3; flag any outliers above 5

**Gotchas:**
- Reference date must be the max date in the *full* dataset, not per-customer — use a single global max before any filtering
- A few customers may have monetary = 0 after exclusions; log and drop them before clustering

---

## Phase 4 — Holdout Construction

**Goal:** Create per-customer ground truth by holding out each customer's most recent invoice for evaluation.

**Inputs:** `transactions` table, `customers` table (for freq_tier labels)

**Steps:**
- [ ] For each customer, identify their most recent invoice by `invoice_date`
- [ ] Extract all `stock_code` values from that invoice as ground truth positives
- [ ] Exclude customers with only one distinct invoice (no training data would remain)
- [ ] Join `freq_tier` from `customers` table
- [ ] Write to `holdout` table
- [ ] Verify all freq_tiers are represented; report counts per tier

**Outputs:** `holdout` table

**Validation:**
- [ ] No customer should appear more than once in `holdout` (one held-out invoice per customer)
- [ ] All five freq_tiers must have at least one customer represented; warn if any tier has fewer than 5
- [ ] Holdout customer count should be < total customer count (some dropped due to single invoice)

**Gotchas:**
- Some invoices contain only one product — that's fine, it's still a valid ground truth
- Ties in invoice_date (same customer, same date, two invoices): pick the one with the higher invoice_id as a tiebreaker

---

## Phase 5 — Collaborative Filtering

**Goal:** Train an ALS collaborative filter as the non-LLM baseline and generate top-K recommendations per customer.

**Inputs:** `transactions` table (excluding holdout invoices), `holdout` table

**Steps:**
- [ ] Filter `transactions` to exclude each customer's holdout invoice
- [ ] Build a sparse customer × product interaction matrix using `scipy.sparse.csr_matrix` (values = quantity, treat as implicit feedback)
- [ ] Fit `implicit.als.AlternatingLeastSquares(factors=50, iterations=20, random_state=random_seed)`
- [ ] For each customer, generate top-K recommendations (K from project config), excluding already-purchased products
- [ ] Write to `cf_recommendations` table with columns: `customer_id`, `product_id`, `rank`, `score`
- [ ] Evaluate against `holdout`: compute HR@K, NDCG@K per customer, coverage across all customers
- [ ] Write per-customer rows to `eval_results` (model=`cf`, tier=`baseline`)
- [ ] Aggregate and write summary row to `run_log`

**Outputs:** `cf_recommendations` table, rows in `eval_results` and `run_log`

**Validation:**
- [ ] CF HR@K for cold customers should be near 0 (expected — they have little signal)
- [ ] CF HR@K for champion customers should be meaningfully above 0
- [ ] Coverage should be > 1% of catalog (sanity check that diverse products are being recommended)

**Gotchas:**
- `implicit` expects items × users matrix, not users × items — transpose before fitting
- Must zero out already-purchased items in recommendations; `implicit` has a `filter_already_liked_items` flag

---

## Phase 6 — LLM Runs (T0–T3)

**Goal:** Generate recommendations at each context enrichment tier using the LLM, checkpointed for resumability.

**Inputs:** `transactions`, `customers`, `products`, `cf_recommendations` tables; for T3: churn/propensity scores computed inline

**Pre-T3 setup:**
- [ ] Train XGBoost churn classifier on customer RFM features; store scores in memory
- [ ] Train XGBoost propensity scorer; store scores in memory

**For each tier T0 → T1 → T2 → T3:**
- [ ] Load customers to process; filter out those already in `llm_results` for this tier + run_id (checkpoint)
- [ ] For each customer, assemble the tier-appropriate context block (see `skills/llm_prompts.md`)
- [ ] Call the API using `model_id` (full run) or `dev_model_id` (smoke test)
- [ ] Parse JSON response; on failure log the raw response and insert a null row — do not abort
- [ ] Write result to `llm_results` immediately after each call
- [ ] Log progress to console every `checkpoint_every` calls

**Outputs:** `llm_results` table (one row per customer per tier)

**Validation:**
- [ ] After each tier completes, verify row count equals expected customer count
- [ ] Parse failure rate should be < 5%; investigate prompt if higher
- [ ] Spot-check 3–5 rationale strings to confirm the model is using the context

**Gotchas:**
- Run tiers in strict T0 → T1 → T2 → T3 order; each tier's prompt builds on the previous context block
- Smoke test uses `dev_model_id` (Haiku) and `sample_size` customers — verify both are set before starting
- T3 churn/propensity scores must be computed before the T3 loop, not inside it

---

## Phase 7 — Evaluation

**Goal:** Compute HR@K, NDCG@K, and coverage for all models and tiers; stratify by frequency tier and RFM segment.

**Inputs:** `holdout`, `cf_recommendations`, `llm_results`, `customers` tables

**Steps:**
- [ ] For each (model, tier) pair — CF baseline + T0/T1/T2/T3:
  - [ ] Join recommendations with holdout per customer
  - [ ] Compute HR@K: 1 if any holdout product appears in top-K, else 0
  - [ ] Compute NDCG@K: position-discounted relevance score
  - [ ] Write one row per customer to `eval_results`
- [ ] Compute coverage: unique products recommended / total products in catalog
- [ ] Aggregate means by freq_tier and by RFM segment
- [ ] Write one summary row per (model, tier) to `run_log`

**Outputs:** `eval_results` table, `run_log` summary rows

**Validation:**
- [ ] T0 should beat CF for cold customers (LLM uses description semantics, CF has no signal)
- [ ] T3 should have highest HR@K overall
- [ ] Coverage should increase with higher context tiers

**Gotchas:**
- LLM recommendations are returned as product descriptions, not stock codes — must fuzzy-match or normalize to stock codes before comparing with holdout
- Customers with no recommendations (parse failure rows) should be excluded from metric aggregation and noted in run_log

---

## Phase 8 — Reporting

**Goal:** Aggregate and visualize benchmark results for comparison across models, tiers, and customer segments.

**Inputs:** `eval_results`, `run_log`, `customers` tables

**Steps:**
- [ ] Pivot table: rows = freq_tier, columns = (model, tier), values = mean HR@K — print to console
- [ ] Same pivot for NDCG@K
- [ ] Bar chart: HR@K by context tier, grouped by freq_tier — save to `data/plots/hr_by_tier.png`
- [ ] Heatmap: RFM segment × context tier, values = HR@K — save to `data/plots/heatmap_segment_tier.png`
- [ ] Print top-line findings: which tier wins overall, which tier wins for cold customers

**Outputs:** Console summary, plots in `data/plots/`

**Validation:**
- [ ] Confirm plot files exist at expected paths after the run
- [ ] Pivot table should show monotonic improvement T0 → T3 for at least cold and sparse tiers

**Gotchas:**
- `data/plots/` directory must be created if it doesn't exist before saving plots
- Use non-interactive matplotlib backend (`matplotlib.use("Agg")`) so plots save without a display

---

## Phase 9 — NL Query Interface

**Goal:** Interactive natural language interface for querying `retail.db` without writing SQL.

**Inputs:** `retail.db` (all tables), schema description string

**Steps:**
- [ ] Build a schema context string from the table definitions (see `skills/metadata.md`)
- [ ] Start a REPL loop: accept a plain-English question from stdin
- [ ] Send schema + question to the LLM with a prompt requesting a single SELECT statement
- [ ] Validate the returned SQL (reject anything that isn't SELECT)
- [ ] Execute against `retail.db`, fetch results
- [ ] Send result rows back to the LLM to format as a natural language answer
- [ ] Print the answer; loop back to prompt
- [ ] Exit cleanly on empty input or `quit`

**Outputs:** Interactive terminal session

**Validation:**
- [ ] Test the four example queries from `skills/metadata.md`
- [ ] Confirm SQL injection guard rejects a `DROP TABLE` attempt

**Gotchas:**
- Result sets can be large — cap at 100 rows before passing back to the LLM for formatting
- Schema context string must stay under ~2000 tokens; use abbreviated column descriptions if needed

---

## Phase 10 — Recommendations API

**Goal:** Serve pre-computed LLM recommendations from `llm_results` via a FastAPI REST API.

**Inputs:** `llm_results`, `products`, `customers` tables

**Steps:**
- [ ] Add `fastapi` and `uvicorn` to `requirements.txt`
- [ ] On startup, check which customer_ids have rows in `llm_results` — these are the available customers
- [ ] Implement `GET /recommendations/{customer_id}` — returns top-K recommendations for the best available tier
  - [ ] Accept optional `?tier=T0|T1|T2|T3` query param; default to highest tier available for that customer
  - [ ] Join product descriptions from `products` table so response includes human-readable names
  - [ ] Return 404 if customer_id has no pre-computed results
- [ ] Implement `GET /customers` — returns list of customer_ids that have pre-computed results
- [ ] Implement `GET /health` — returns 200 OK with db row counts for monitoring

**Response shape (`/recommendations/{customer_id}`):**
```json
{
  "customer_id": "12345",
  "tier": "T3",
  "recommendations": [
    {"rank": 1, "product_id": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER"},
    ...
  ],
  "rationale": "..."
}
```

**Outputs:** FastAPI server running on `http://localhost:8000`

**Validation:**
- [ ] `GET /health` returns 200 and correct row counts
- [ ] `GET /recommendations/{customer_id}` returns correct tier and K results for a known customer
- [ ] `GET /recommendations/UNKNOWN` returns 404
- [ ] `GET /recommendations/{customer_id}?tier=T0` returns T0 results even if T3 is available
- [ ] `GET /customers` returns only customer_ids present in `llm_results`

**Gotchas:**
- Not all customers will have pre-computed results if Phase 6 ran on a subset — the API should surface only what's available, not error
- Open a new db connection per request (SQLite doesn't support shared connections across threads safely)
- Phase 10 is read-only — it never writes to the database
