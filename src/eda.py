"""
Phase 2 — EDA
Profile the transactions table and save key figures and tables to outputs/eda/.
"""

from __future__ import annotations

import os
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.ingest import RAW_XLSX

DB_PATH = "data/retail.db"
EDA_DIR = "outputs/eda"
CANCELLATION_PREFIX = "C"

FREQ_TIERS = [
    ("cold", 1, 2),
    ("sparse", 3, 5),
    ("moderate", 6, 15),
    ("rich", 16, 30),
    ("champion", 31, None),
]


def assign_freq_tier(n: int) -> str:
    for name, lo, hi in FREQ_TIERS:
        if n >= lo and (hi is None or n <= hi):
            return name
    return "unknown"


def quality_report(con: sqlite3.Connection, out_path: str) -> None:
    df = pd.read_sql("SELECT * FROM transactions", con)
    lines = []
    lines.append(f"shape: {df.shape}")
    lines.append("")
    lines.append("dtypes:")
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("null counts:")
    lines.append(df.isna().sum().to_string())
    lines.append("")
    exact_dupes = int(df.duplicated().sum())
    invoice_product_dupes = int(df.duplicated(subset=["invoice_id", "stock_code"]).sum())
    lines.append(f"exact duplicate rows: {exact_dupes}")
    lines.append(f"duplicate (invoice_id, stock_code) rows: {invoice_product_dupes}")
    lines.append("")
    neg_qty = int((df["quantity"] <= 0).sum())
    neg_price = int((df["price"] <= 0).sum())
    lines.append(f"rows with quantity <= 0 (post-clean, expect 0): {neg_qty}")
    lines.append(f"rows with price <= 0 (post-clean, expect 0): {neg_price}")
    lines.append("")
    lines.append(f"date range: {df['invoice_date'].min()} -> {df['invoice_date'].max()}")
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {out_path}")
    print(f"duplicate (invoice_id, stock_code) count: {invoice_product_dupes}")


def freq_tier_counts(con: sqlite3.Connection, out_path: str) -> pd.DataFrame:
    counts = pd.read_sql(
        """
        SELECT customer_id, COUNT(DISTINCT invoice_id) AS n_invoices
        FROM transactions
        GROUP BY customer_id
        """,
        con,
    )
    counts["freq_tier"] = counts["n_invoices"].apply(assign_freq_tier)
    tier_counts = (
        counts.groupby("freq_tier")
        .size()
        .reindex([t[0] for t in FREQ_TIERS], fill_value=0)
        .reset_index(name="customer_count")
    )
    total = tier_counts["customer_count"].sum()
    tier_counts["pct"] = (tier_counts["customer_count"] / total * 100).round(2)
    tier_counts.to_csv(out_path, index=False)
    print("\nfrequency tier distribution:")
    print(tier_counts.to_string(index=False))
    for _, row in tier_counts.iterrows():
        if row["customer_count"] < 10:
            print(f"  WARNING: tier '{row['freq_tier']}' has only {row['customer_count']} customers")
    cold_sparse_pct = tier_counts[tier_counts["freq_tier"].isin(["cold", "sparse"])]["pct"].sum()
    print(f"  cold + sparse share: {cold_sparse_pct:.1f}%")
    return tier_counts


def top_products(con: sqlite3.Connection, out_path: str) -> None:
    top20 = pd.read_sql(
        """
        SELECT stock_code, MAX(description) AS description, COUNT(*) AS transaction_count
        FROM transactions
        GROUP BY stock_code
        ORDER BY transaction_count DESC
        LIMIT 20
        """,
        con,
    )
    top20.to_csv(out_path, index=False)
    print("\ntop-20 products by transaction volume:")
    print(top20.to_string(index=False))
    single_tx = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT stock_code FROM transactions
            GROUP BY stock_code HAVING COUNT(*) = 1
        )
        """
    ).fetchone()[0]
    print(f"\nproducts with only one transaction: {single_tx}")


def monthly_volume(con: sqlite3.Connection, out_path: str) -> None:
    df = pd.read_sql(
        "SELECT invoice_date FROM transactions",
        con,
        parse_dates=["invoice_date"],
    )
    monthly = df.set_index("invoice_date").resample("MS").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_title("Monthly transaction volume")
    ax.set_xlabel("Month")
    ax.set_ylabel("Transaction count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nwrote {out_path}")


def cancellation_by_country(out_path: str) -> None:
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None)
    raw = pd.concat(sheets.values(), ignore_index=True)
    raw["Invoice"] = raw["Invoice"].astype(str)
    raw["is_cancellation"] = raw["Invoice"].str.startswith(CANCELLATION_PREFIX)
    by_country = (
        raw.groupby("Country")
        .agg(total=("Invoice", "size"), cancellations=("is_cancellation", "sum"))
        .reset_index()
    )
    by_country["cancellation_rate"] = (by_country["cancellations"] / by_country["total"]).round(4)
    by_country = by_country.sort_values("total", ascending=False)
    by_country.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(by_country.head(10).to_string(index=False))


if __name__ == "__main__":
    os.makedirs(EDA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        quality_report(con, os.path.join(EDA_DIR, "quality_report.txt"))
        freq_tier_counts(con, os.path.join(EDA_DIR, "freq_tier_counts.csv"))
        top_products(con, os.path.join(EDA_DIR, "top_products.csv"))
        monthly_volume(con, os.path.join(EDA_DIR, "monthly_volume.png"))
        cancellation_by_country(os.path.join(EDA_DIR, "cancellation_by_country.csv"))
    finally:
        con.close()
