# Skill: Database

## Purpose
Design, create, and query the SQLite database that holds all data, features, model
outputs, and evaluation results for this project.

## Required Project Config
| Key | Used for |
|---|---|
| `db_path` | SQLite connection path |

## Connection pattern
```python
import sqlite3
con = sqlite3.connect(db_path)  # db_path from project config
con.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
```

## Schema creation
Run CREATE TABLE IF NOT EXISTS for all tables at startup so scripts are idempotent.
Never DROP and recreate — use INSERT OR REPLACE or INSERT OR IGNORE to upsert.

## Table responsibilities

| Table | Written by | Key columns |
|---|---|---|
| transactions | ingest.py | invoice_id, customer_id, product_id, quantity, price, invoice_date |
| customers | features.py | customer_id, recency, frequency, monetary, segment, freq_tier |
| products | features.py | product_id, description, popularity_rank, seasonality_index |
| holdout | holdout.py | customer_id, product_id, invoice_date |
| cf_recommendations | cf.py | customer_id, product_id, rank, score |
| llm_results | llm.py | customer_id, tier, recommendations (JSON), rationale, model, run_id |
| eval_results | evaluate.py | customer_id, model, tier, hr_at_k, ndcg_at_k, k |
| run_log | any script | run_id, script, model, tier, params (JSON), hr_mean, ndcg_mean, coverage, ts |

## Checkpointing pattern
Before inserting an LLM result, check if (customer_id, tier, run_id) already exists.
Skip if present. This makes runs resumable after interruption.

```python
existing = pd.read_sql(
    "SELECT customer_id FROM llm_results WHERE tier=? AND run_id=?",
    con, params=(tier, run_id)
)
remaining = all_customers[~all_customers.isin(existing.customer_id)]
```

## Logging runs
Every script that produces model output must insert a row into run_log on completion:
- run_id: uuid4
- params: json.dumps of all relevant hyperparameters
- ts: datetime.utcnow().isoformat()
