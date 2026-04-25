# Skill: EDA

## Purpose
Profile a transactional retail dataset, surface data quality issues, and characterize
customer and product distributions before feature engineering.

## Required Project Config
| Key | Used for |
|---|---|
| `cancellation_prefix` | Identifying and auditing cancellation invoices |
| frequency tier thresholds | Assigning tier labels to customers |

## Steps

### 1. Load and inspect
- Check shape, dtypes, and memory usage
- Print head + sample rows to confirm field semantics

### 2. Quality checks
- Count nulls per column; flag columns exceeding 5% null rate
- Count duplicate rows (exact) and duplicate invoice+product combinations
- Verify date range is contiguous and plausible
- Flag negative quantities and negative/zero prices as anomalies

### 3. Cancellation audit
- Identify cancellation marker from project config
- Count cancellations vs normal invoices; compute cancellation rate by country and by month
- Do NOT drop cancellations yet — report counts first, then exclude for all downstream work

### 4. Customer activity distribution
- Compute purchase count per customer (excluding cancellations)
- Assign frequency tier labels using thresholds from project config
- Report tier counts and % of customer base per tier

### 5. Product distribution
- Count distinct products; report top-20 by transaction volume
- Flag products with only one transaction (long tail)

### 6. Temporal patterns
- Monthly transaction volume plot
- Identify any obvious seasonal spikes worth flagging for feature engineering

## Output
- Printed summary stats (no separate report file needed)
- Frequency tier counts logged to console for verification before moving to features phase
