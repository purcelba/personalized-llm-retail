"""
Phase 4 — Holdout Construction
Hold out each customer's most recent invoice as ground truth for evaluation.
Writes to the holdout table.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.eda import FREQ_TIERS

DB_PATH = "data/retail.db"


def build_holdout(con: sqlite3.Connection) -> pd.DataFrame:
    tx = pd.read_sql(
        "SELECT customer_id, invoice_id, stock_code, invoice_date FROM transactions",
        con,
        parse_dates=["invoice_date"],
    )
    invoice_counts = tx.groupby("customer_id")["invoice_id"].nunique()
    eligible = invoice_counts[invoice_counts > 1].index
    print(f"customers with >1 invoice (eligible for holdout): {len(eligible)}")
    print(f"customers excluded (single invoice): {(invoice_counts == 1).sum()}")
    tx = tx[tx["customer_id"].isin(eligible)]

    inv_meta = (
        tx.groupby(["customer_id", "invoice_id"])["invoice_date"]
        .max()
        .reset_index()
    )
    inv_meta = inv_meta.sort_values(
        ["customer_id", "invoice_date", "invoice_id"], ascending=[True, False, False]
    )
    last_inv = inv_meta.drop_duplicates("customer_id", keep="first")[
        ["customer_id", "invoice_id"]
    ]

    holdout = tx.merge(last_inv, on=["customer_id", "invoice_id"], how="inner")
    holdout = holdout[["customer_id", "invoice_id", "stock_code", "invoice_date"]].copy()
    holdout["invoice_date"] = holdout["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    holdout = holdout.drop_duplicates(subset=["customer_id", "stock_code"])

    customers = pd.read_sql("SELECT customer_id, freq_tier FROM customers", con)
    holdout = holdout.merge(customers, on="customer_id", how="left")

    return holdout


def load_holdout(df: pd.DataFrame, con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS holdout (
            customer_id   TEXT,
            invoice_id    TEXT,
            stock_code    TEXT,
            invoice_date  TEXT,
            freq_tier     TEXT,
            PRIMARY KEY (customer_id, stock_code)
        )
        """
    )
    con.execute("DELETE FROM holdout")
    df.to_sql("holdout", con, if_exists="append", index=False)
    con.commit()


def validate_holdout(con: sqlite3.Connection) -> None:
    n_customers = con.execute("SELECT COUNT(DISTINCT customer_id) FROM holdout").fetchone()[0]
    n_rows = con.execute("SELECT COUNT(*) FROM holdout").fetchone()[0]
    n_invoices = con.execute("SELECT COUNT(DISTINCT invoice_id || '|' || customer_id) FROM holdout").fetchone()[0]
    print(f"\nholdout rows: {n_rows}")
    print(f"customers in holdout: {n_customers}")
    print(f"distinct (customer, invoice) pairs: {n_invoices}")
    assert n_invoices == n_customers, "each customer must have exactly one held-out invoice"

    total_customers = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    print(f"total customers in db: {total_customers} (holdout < total expected)")
    assert n_customers < total_customers

    print("\nholdout customers per freq_tier:")
    tiers = pd.read_sql(
        "SELECT freq_tier, COUNT(DISTINCT customer_id) AS n FROM holdout GROUP BY freq_tier",
        con,
    ).set_index("freq_tier")["n"]
    tiers = tiers.reindex([t[0] for t in FREQ_TIERS], fill_value=0)
    print(tiers.to_string())
    for tier, n in tiers.items():
        if n == 0:
            print(f"  ERROR: tier '{tier}' has zero customers in holdout")
        elif n < 5:
            print(f"  WARNING: tier '{tier}' has only {n} customers")


if __name__ == "__main__":
    con = sqlite3.connect(DB_PATH)
    try:
        holdout = build_holdout(con)
        load_holdout(holdout, con)
        validate_holdout(con)
    finally:
        con.close()
