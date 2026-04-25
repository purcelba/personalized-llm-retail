# Skill: Evaluation

## Purpose
Construct holdout sets and compute ranking metrics to compare CF baseline against
LLM recommendation tiers.

## Required Project Config
| Key | Used for |
|---|---|
| `top_k` | K value for HR@K, NDCG@K |
| frequency tier thresholds | Stratifying metric reports by tier |

## Holdout construction (→ holdout table)
- For each customer, hold out their single most recent invoice (by date)
- All products on that invoice are ground truth positives
- Customers with only one invoice are excluded from evaluation (nothing left to train on)
- Verify holdout covers all frequency tiers — report counts per tier before proceeding

## Metrics

### Hit Rate @ K (HR@K)
```
HR@K = 1 if any holdout item appears in top-K recommendations else 0
```
Aggregate: mean across customers, reported per frequency tier and RFM segment.

### NDCG @ K
```
DCG  = sum(rel_i / log2(i+1)) for i in 1..K  (rel_i = 1 if item is holdout, else 0)
IDCG = DCG of perfect ranking
NDCG = DCG / IDCG
```
Aggregate: mean across customers, reported per frequency tier and RFM segment.

### Coverage
```
coverage = |unique products recommended| / |total products in catalog|
```
Compute across all customers for a given model+tier combination.

## Stratification
Always report metrics broken down by:
1. Frequency tier (from project config thresholds)
2. RFM segment label

This reveals whether LLM context tiers help cold/sparse users more than rich/champion users.

## K value
Use K from project config. Compute and store results for that K only unless instructed otherwise.

## Writing results
Insert one row per (customer_id, model, tier) into eval_results.
After all customers are evaluated, insert aggregate metrics into run_log.
