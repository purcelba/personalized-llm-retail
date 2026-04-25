# Skill: Metadata & Natural Language Query Interface

## Purpose
Enrich stored outputs with human-readable labels and build a natural language interface
for querying the database without writing SQL.

## Required Project Config
| Key | Used for |
|---|---|
| `db_path` | Database connection and schema context string |
| `model_id` | LLM used to generate SQL and format answers |

## Output enrichment
Before building the NL interface, verify these columns exist in the database:
- `customers.segment` — descriptive label (not just a cluster number)
- `customers.freq_tier` — cold/sparse/moderate/rich/champion
- `products.description` — human-readable name, not just product ID
- `eval_results.tier` — T0/T1/T2/T3 label
- `run_log.ts` — ISO timestamp

If any are missing, backfill from the raw data before proceeding.

## NL query interface design
The interface takes a plain-English question, converts it to SQL, executes against the
database, and returns a formatted answer.

### Approach
1. Build a schema context string describing all tables and key columns
2. Pass schema + user question to the LLM with a prompt asking for a single SQL query
3. Execute the query against the database
4. Pass the result rows back to the LLM to format as a natural language answer

### Schema context template
```
Database: {db_path}
Tables:
- transactions(invoice_id, customer_id, product_id, quantity, price, invoice_date)
- customers(customer_id, recency, frequency, monetary, segment, freq_tier)
- products(product_id, description, popularity_rank, seasonality_index)
- holdout(customer_id, product_id, invoice_date)
- cf_recommendations(customer_id, product_id, rank, score)
- llm_results(customer_id, tier, recommendations, rationale, model, run_id)
- eval_results(customer_id, model, tier, hr_at_k, ndcg_at_k, k)
- run_log(run_id, script, model, tier, params, hr_mean, ndcg_mean, coverage, ts)
```

### Safety
- Only allow SELECT statements — reject any query containing INSERT, UPDATE, DELETE, DROP
- Limit result rows to 100 unless user specifies otherwise

## Example queries the interface should handle
- "Which context tier had the highest HR@10 for cold customers?"
- "Show me the top 5 most recommended products in T3"
- "How does NDCG compare between champion and cold segments for T2?"
- "Which customers had a hit in T0 but not in CF?"
