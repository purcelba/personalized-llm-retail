"""
Phase 5 — Collaborative Filtering
Train an ALS model and generate top-K recommendations as the non-LLM baseline.
Writes to the cf_recommendations table.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DB_PATH = "data/retail.db"
RANDOM_SEED = 42
TOP_K = 10
ALS_FACTORS = 50
ALS_ITERATIONS = 20
ALS_REGULARIZATION = 0.01
ALS_ALPHA = 40.0  # confidence multiplier on quantity

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def build_interaction_matrix(con: sqlite3.Connection):
    holdout = pd.read_sql(
        "SELECT customer_id, invoice_id FROM holdout", con
    ).drop_duplicates()
    holdout_pairs = set(zip(holdout["customer_id"], holdout["invoice_id"]))

    tx = pd.read_sql(
        "SELECT customer_id, invoice_id, stock_code, quantity FROM transactions",
        con,
    )
    keep = ~tx.apply(
        lambda r: (r["customer_id"], r["invoice_id"]) in holdout_pairs, axis=1
    )
    tx = tx[keep]

    interactions = (
        tx.groupby(["customer_id", "stock_code"])["quantity"]
        .sum()
        .reset_index()
    )
    interactions = interactions[interactions["quantity"] > 0]

    customers = sorted(interactions["customer_id"].unique())
    products = sorted(interactions["stock_code"].unique())
    cust_idx = {c: i for i, c in enumerate(customers)}
    prod_idx = {p: i for i, p in enumerate(products)}

    rows = interactions["customer_id"].map(cust_idx).values
    cols = interactions["stock_code"].map(prod_idx).values
    vals = interactions["quantity"].astype(np.float32).values

    matrix = csr_matrix(
        (vals, (rows, cols)), shape=(len(customers), len(products))
    )
    print(
        f"interaction matrix: {matrix.shape[0]} customers x {matrix.shape[1]} products, "
        f"{matrix.nnz} nonzero entries (density {matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.4%})"
    )
    return matrix, customers, products


def train_als(matrix: csr_matrix) -> implicit.als.AlternatingLeastSquares:
    confidence = matrix.copy()
    confidence.data = 1.0 + ALS_ALPHA * confidence.data
    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION,
        random_state=RANDOM_SEED,
        use_gpu=False,
    )
    model.fit(confidence)
    return model


def generate_recommendations(
    model, matrix: csr_matrix, customers: list[str], products: list[str]
) -> pd.DataFrame:
    rows = []
    n = len(customers)
    ids, scores = model.recommend(
        np.arange(n),
        matrix,
        N=TOP_K,
        filter_already_liked_items=True,
    )
    for i, cid in enumerate(customers):
        for rank, (j, score) in enumerate(zip(ids[i], scores[i]), start=1):
            if j < 0:
                continue
            rows.append((cid, products[j], rank, float(score)))
    df = pd.DataFrame(rows, columns=["customer_id", "stock_code", "rank", "score"])
    print(f"generated {len(df)} recommendation rows for {df['customer_id'].nunique()} customers")
    return df


def write_recommendations(df: pd.DataFrame, con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cf_recommendations (
            customer_id TEXT,
            stock_code  TEXT,
            rank        INTEGER,
            score       REAL,
            PRIMARY KEY (customer_id, rank)
        )
        """
    )
    con.execute("DELETE FROM cf_recommendations")
    df.to_sql("cf_recommendations", con, if_exists="append", index=False)
    con.commit()


def evaluate(con: sqlite3.Connection, run_id: str) -> dict:
    holdout = pd.read_sql(
        "SELECT customer_id, stock_code, freq_tier FROM holdout", con
    )
    recs = pd.read_sql(
        "SELECT customer_id, stock_code, rank FROM cf_recommendations", con
    )
    customers = pd.read_sql(
        "SELECT customer_id, freq_tier, segment FROM customers", con
    )

    holdout_grp = holdout.groupby("customer_id")["stock_code"].apply(set)
    recs_grp = (
        recs.sort_values(["customer_id", "rank"])
        .groupby("customer_id")
        .apply(lambda g: list(zip(g["stock_code"], g["rank"])), include_groups=False)
    )

    per_customer = []
    for cid, true_set in holdout_grp.items():
        ranked = recs_grp.get(cid, [])
        hit = 0
        dcg = 0.0
        for sc, r in ranked[:TOP_K]:
            if sc in true_set:
                hit = 1
                dcg += 1.0 / math.log2(r + 1)
        n_rel = min(len(true_set), TOP_K)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1)) if n_rel > 0 else 1.0
        ndcg = dcg / idcg if idcg > 0 else 0.0
        per_customer.append(
            {
                "customer_id": cid,
                "model": "cf",
                "tier": "baseline",
                "k": TOP_K,
                "hr_at_k": hit,
                "ndcg_at_k": ndcg,
                "run_id": run_id,
            }
        )
    eval_df = pd.DataFrame(per_customer).merge(customers, on="customer_id", how="left")

    catalog_size = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    coverage = recs["stock_code"].nunique() / catalog_size

    print(f"\noverall HR@{TOP_K}: {eval_df['hr_at_k'].mean():.4f}")
    print(f"overall NDCG@{TOP_K}: {eval_df['ndcg_at_k'].mean():.4f}")
    print(f"coverage: {coverage:.4%} ({recs['stock_code'].nunique()}/{catalog_size} products)")

    print("\nHR@K by freq_tier:")
    print(
        eval_df.groupby("freq_tier")["hr_at_k"].agg(["mean", "count"]).round(4).to_string()
    )

    return {
        "eval_df": eval_df.drop(columns=["segment"]),
        "hr_mean": float(eval_df["hr_at_k"].mean()),
        "ndcg_mean": float(eval_df["ndcg_at_k"].mean()),
        "coverage": float(coverage),
    }


def write_eval_results(eval_df: pd.DataFrame, con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_results (
            customer_id TEXT,
            model       TEXT,
            tier        TEXT,
            k           INTEGER,
            hr_at_k     REAL,
            ndcg_at_k   REAL,
            freq_tier   TEXT,
            run_id      TEXT,
            PRIMARY KEY (customer_id, model, tier, run_id)
        )
        """
    )
    con.execute(
        "DELETE FROM eval_results WHERE model='cf' AND tier='baseline'"
    )
    eval_df.to_sql("eval_results", con, if_exists="append", index=False)
    con.commit()


def write_run_log(
    con: sqlite3.Connection,
    run_id: str,
    metrics: dict,
    n_customers: int,
) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS run_log (
            run_id      TEXT,
            script      TEXT,
            model       TEXT,
            tier        TEXT,
            params      TEXT,
            n_customers INTEGER,
            hr_mean     REAL,
            ndcg_mean   REAL,
            coverage    REAL,
            ts          TEXT
        )
        """
    )
    params = {
        "factors": ALS_FACTORS,
        "iterations": ALS_ITERATIONS,
        "regularization": ALS_REGULARIZATION,
        "alpha": ALS_ALPHA,
        "top_k": TOP_K,
        "random_seed": RANDOM_SEED,
    }
    con.execute(
        """
        INSERT INTO run_log
        (run_id, script, model, tier, params, n_customers, hr_mean, ndcg_mean, coverage, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            "cf.py",
            "cf",
            "baseline",
            json.dumps(params),
            n_customers,
            metrics["hr_mean"],
            metrics["ndcg_mean"],
            metrics["coverage"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    con.commit()


if __name__ == "__main__":
    run_id = str(uuid.uuid4())
    print(f"run_id: {run_id}")
    con = sqlite3.connect(DB_PATH)
    try:
        matrix, customers, products = build_interaction_matrix(con)
        model = train_als(matrix)
        recs = generate_recommendations(model, matrix, customers, products)
        write_recommendations(recs, con)
        metrics = evaluate(con, run_id)
        write_eval_results(metrics["eval_df"], con)
        write_run_log(con, run_id, metrics, n_customers=len(metrics["eval_df"]))
        print(f"\nrun_log written for run_id={run_id}")
    finally:
        con.close()
