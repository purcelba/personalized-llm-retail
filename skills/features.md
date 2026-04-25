# Skill: Feature Engineering

## Purpose
Build customer-level RFM features, product-level popularity and seasonality features,
and co-occurrence signal for collaborative filtering neighbors.

## Required Project Config
| Key | Used for |
|---|---|
| `cancellation_prefix` | Excluding cancellations from frequency and monetary calculations |
| `random_seed` | KMeans and any other random state |
| `kmeans_k` | Number of RFM segments |
| `seasonality_cv_threshold` | Cutoff for flagging a product as highly seasonal |
| frequency tier thresholds | Assigning freq_tier labels to customers |

## Customer features (→ customers table)

### RFM
- **Recency**: days between reference date (max invoice date in dataset) and customer's last purchase
- **Frequency**: count of distinct invoices (cancellations excluded per project config)
- **Monetary**: sum of (quantity × price) across all non-cancelled invoices
- Log-transform all three before clustering to reduce skew

### Segmentation
- Standardize RFM with StandardScaler
- Fit KMeans with k from project config, seed from project config
- Label segments descriptively (e.g. champions, at_risk, new) based on centroid ranks
- Add frequency tier label using thresholds from project config

## Product features (→ products table)

### Popularity rank
- Count total units sold per product (cancellations excluded)
- Rank ascending (1 = most popular)
- Store raw count and rank

### Seasonality index
- Aggregate monthly sales per product
- Compute coefficient of variation across months (std / mean)
- High CV → high seasonality; threshold from project config

## Co-occurrence (used by CF skill)
- Build customer × product binary matrix (1 if purchased)
- Not stored as a table — computed in-memory during CF phase

## Notes
- Use seed from project config everywhere random state is needed
- Write customers and products tables; do not overwrite if data already exists — use INSERT OR REPLACE
