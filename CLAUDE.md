# Retail Personalization — LLM Recommendation Benchmark

## Goal
Build a personalized product recommendation system using the UCI Online Retail II dataset.
Benchmark LLM recommendation quality across four context enrichment tiers against a
collaborative filtering baseline. Store all outputs in a SQLite database queryable
via natural language.

---

## Dataset
UCI Online Retail II (ucimlrepo id=502)
Fields: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
Key quirk: Invoices starting with 'C' are cancellations — always exclude unless studying them.

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

Natural language queries over the database are the final deliverable of Phase 9.

---

## Benchmark Design

### Frequency tiers
Thresholds defined in Project Config. Stratify all evaluations by these tiers.

### Context tiers (what gets injected into the LLM prompt)
| Tier | Context added                                              |
|------|------------------------------------------------------------|
| T0   | Raw purchase history only                                  |
| T1   | + RFM segment label + global popularity rank              |
| T2   | + CF top-N neighbor products                              |
| T3   | + churn probability + propensity scores + seasonality     |

### Metrics
- Hit Rate @10 — did the held-out item appear in top-10 recommendations?
- NDCG @10 — were the relevant items ranked higher?
- Coverage — what % of catalog was recommended across all customers?
- Report all metrics split by frequency tier and RFM segment.

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
| `skills/metadata.md`    | Enriching outputs, NL query interface            |

---

## Conventions
- Python 3.10+, pandas, scikit-learn, implicit, xgboost, anthropic, sqlite3
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

1. **Smoke test first** — run the full Phase 1–9 pipeline on a small sample using the dev model (see Project Config). The goal is to verify the pipeline is wired correctly end-to-end, not to produce meaningful metrics.
2. **Sign off before scaling** — only proceed to a full run after the smoke test completes without errors and eval_results rows are populated with plausible values.
3. **Full run** — rerun all phases with `sample_size=None` and `model_id` (Sonnet).

When starting any phase, check `docs/phases.md` status. If the smoke test column is not marked `done`, implement and run the smoke test before the full run.

---

## Execution order
See `docs/phases.md` for status and detail on each phase.
