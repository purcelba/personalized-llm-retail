"""
Phase 1 — Ingest & Clean
Download UCI Online Retail II, clean, and load into the transactions table.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd

DATASET_ID = 502
DB_PATH = "data/retail.db"
RAW_DIR = "data/raw"
RAW_XLSX = "data/raw/online_retail_II.xlsx"
RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
CANCELLATION_PREFIX = "C"

COLUMN_RENAMES = {
    "Invoice": "invoice_id",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def download_raw() -> pd.DataFrame:
    # ucimlrepo lists id=502 but does not serve it; fall back to UCI archive xlsx.
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(RAW_XLSX):
        import urllib.request
        print(f"downloading {RAW_URL}")
        urllib.request.urlretrieve(RAW_URL, RAW_XLSX)
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_RENAMES).copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["invoice_id"] = df["invoice_id"].astype(str)

    before = len(df)
    df = df[df["customer_id"].notna()]
    df = df[~df["invoice_id"].str.startswith(CANCELLATION_PREFIX)]
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    after = len(df)

    df["customer_id"] = df["customer_id"].astype(int).astype(str)
    df["stock_code"] = df["stock_code"].astype(str)

    print(f"rows before clean: {before}")
    print(f"rows after clean:  {after}")
    return df.reset_index(drop=True)


def load(df: pd.DataFrame, con: sqlite3.Connection) -> None:
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            invoice_id    TEXT,
            stock_code    TEXT,
            description   TEXT,
            quantity      INTEGER,
            invoice_date  TEXT,
            price         REAL,
            customer_id   TEXT,
            country       TEXT
        )
        """
    )
    con.execute("DELETE FROM transactions")
    df_to_write = df.copy()
    df_to_write["invoice_date"] = df_to_write["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_to_write.to_sql("transactions", con, if_exists="append", index=False)
    con.commit()


def validate(con: sqlite3.Connection) -> None:
    null_customers = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE customer_id IS NULL OR customer_id = ''"
    ).fetchone()[0]
    assert null_customers == 0, f"found {null_customers} null customer_id rows"

    cancellations = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE invoice_id LIKE 'C%'"
    ).fetchone()[0]
    assert cancellations == 0, f"found {cancellations} cancellation rows"

    dmin, dmax = con.execute(
        "SELECT MIN(invoice_date), MAX(invoice_date) FROM transactions"
    ).fetchone()
    print(f"date range: {dmin} -> {dmax}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    raw = download_raw()
    cleaned = clean(raw)
    con = sqlite3.connect(DB_PATH)
    try:
        load(cleaned, con)
        validate(con)
    finally:
        con.close()
