"""
Phase 3 — Feature Engineering
Build RFM customer features and product popularity/seasonality features.
Writes to the customers and products tables.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.eda import FREQ_TIERS, assign_freq_tier

DB_PATH = "data/retail.db"
RANDOM_SEED = 42
KMEANS_K = 4
SEASONALITY_CV_THRESHOLD = 0.5


def compute_rfm(con: sqlite3.Connection) -> pd.DataFrame:
    tx = pd.read_sql(
        "SELECT customer_id, invoice_id, invoice_date, quantity, price FROM transactions",
        con,
        parse_dates=["invoice_date"],
    )
    ref_date = tx["invoice_date"].max()
    tx["revenue"] = tx["quantity"] * tx["price"]

    rfm = (
        tx.groupby("customer_id")
        .agg(
            last_purchase=("invoice_date", "max"),
            frequency=("invoice_id", "nunique"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (ref_date - rfm["last_purchase"]).dt.days
    rfm = rfm.drop(columns=["last_purchase"])

    dropped = int((rfm["monetary"] <= 0).sum())
    if dropped > 0:
        print(f"dropping {dropped} customers with monetary <= 0")
    rfm = rfm[rfm["monetary"] > 0].reset_index(drop=True)

    rfm["freq_tier"] = rfm["frequency"].apply(assign_freq_tier)
    return rfm[["customer_id", "recency", "frequency", "monetary", "freq_tier"]]


def segment_customers(rfm: pd.DataFrame) -> pd.DataFrame:
    feats = rfm[["recency", "frequency", "monetary"]].copy()
    feats_log = np.log1p(feats)
    scaled = StandardScaler().fit_transform(feats_log)

    km = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(scaled)

    centroids = pd.DataFrame(km.cluster_centers_, columns=["recency_z", "frequency_z", "monetary_z"])
    # Score each centroid: high freq+monetary, low recency = best (champions).
    centroids["score"] = (
        -centroids["recency_z"] + centroids["frequency_z"] + centroids["monetary_z"]
    )
    ranked = centroids.sort_values("score", ascending=False).reset_index().rename(columns={"index": "cluster"})

    names_by_k = {
        4: ["champions", "loyal", "at_risk", "hibernating"],
        3: ["champions", "loyal", "at_risk"],
        5: ["champions", "loyal", "potential", "at_risk", "hibernating"],
    }
    names = names_by_k.get(KMEANS_K, [f"seg_{i}" for i in range(KMEANS_K)])
    cluster_to_name = dict(zip(ranked["cluster"], names))

    out = rfm.copy()
    out["segment"] = pd.Series(labels).map(cluster_to_name).values
    return out


def compute_product_features(con: sqlite3.Connection) -> pd.DataFrame:
    tx = pd.read_sql(
        "SELECT stock_code, description, quantity, invoice_date FROM transactions",
        con,
        parse_dates=["invoice_date"],
    )
    desc = (
        tx.groupby("stock_code")["description"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    units = tx.groupby("stock_code")["quantity"].sum().reset_index(name="units_sold")
    units["popularity_rank"] = units["units_sold"].rank(method="min", ascending=False).astype(int)

    tx["month"] = tx["invoice_date"].dt.to_period("M")
    monthly = tx.groupby(["stock_code", "month"])["quantity"].sum().reset_index()
    cv = (
        monthly.groupby("stock_code")["quantity"]
        .agg(lambda s: s.std(ddof=0) / s.mean() if s.mean() > 0 else 0.0)
        .reset_index(name="seasonality_index")
    )

    products = desc.merge(units, on="stock_code").merge(cv, on="stock_code")
    products["seasonal_flag"] = (products["seasonality_index"] > SEASONALITY_CV_THRESHOLD).astype(int)
    return products[
        ["stock_code", "description", "units_sold", "popularity_rank", "seasonality_index", "seasonal_flag"]
    ]


def load_customers(df: pd.DataFrame, con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            recency     INTEGER,
            frequency   INTEGER,
            monetary    REAL,
            freq_tier   TEXT,
            segment     TEXT
        )
        """
    )
    con.execute("DELETE FROM customers")
    df.to_sql("customers", con, if_exists="append", index=False)
    con.commit()


def load_products(df: pd.DataFrame, con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            stock_code        TEXT PRIMARY KEY,
            description       TEXT,
            units_sold        INTEGER,
            popularity_rank   INTEGER,
            seasonality_index REAL,
            seasonal_flag     INTEGER
        )
        """
    )
    con.execute("DELETE FROM products")
    df.to_sql("products", con, if_exists="append", index=False)
    con.commit()


def validate(customers: pd.DataFrame, products: pd.DataFrame) -> None:
    seg_counts = customers["segment"].value_counts(normalize=True) * 100
    print("\nsegment distribution:")
    print(seg_counts.round(2).to_string())
    if (seg_counts > 60).any():
        big = seg_counts[seg_counts > 60].index.tolist()
        print(f"  WARNING: segment(s) {big} exceed 60% share")

    print("\nfreq_tier distribution (post-RFM, monetary>0 only):")
    tier_counts = (
        customers["freq_tier"]
        .value_counts()
        .reindex([t[0] for t in FREQ_TIERS], fill_value=0)
    )
    print(tier_counts.to_string())

    s = products["seasonality_index"]
    print(f"\nseasonality_index: min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f}")
    outliers = int((s > 5).sum())
    if outliers:
        print(f"  WARNING: {outliers} products have seasonality_index > 5")
    flagged = int(products["seasonal_flag"].sum())
    print(f"products flagged seasonal (CV > {SEASONALITY_CV_THRESHOLD}): {flagged}")


if __name__ == "__main__":
    con = sqlite3.connect(DB_PATH)
    try:
        rfm = compute_rfm(con)
        customers = segment_customers(rfm)
        products = compute_product_features(con)
        load_customers(customers, con)
        load_products(products, con)
        print(f"loaded {len(customers)} customers, {len(products)} products")
        validate(customers, products)
    finally:
        con.close()
