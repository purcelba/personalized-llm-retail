"""
Phase 7 — Evaluation
Compute HR@K, NDCG@K, and coverage for CF and all LLM tiers.
Stratifies results by frequency tier and RFM segment.
Writes per-customer rows to eval_results and aggregate summaries to run_log.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd

DB_PATH = "data/retail.db"
TOP_K = 10
TIERS = ["llm_base", "llm_cf"]


def _ndcg(ranked: list[str], truth: set[str], k: int) -> float:
    dcg = 0.0
    for r, sc in enumerate(ranked[:k], start=1):
        if sc in truth:
            dcg += 1.0 / math.log2(r + 1)
    n_rel = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1)) if n_rel else 1.0
    return dcg / idcg if idcg > 0 else 0.0


def _hr(ranked: list[str], truth: set[str], k: int) -> int:
    return int(any(sc in truth for sc in ranked[:k]))


def load_recs_cf(con: sqlite3.Connection) -> dict[str, list[str]]:
    df = pd.read_sql(
        "SELECT customer_id, stock_code, rank FROM cf_recommendations ORDER BY customer_id, rank",
        con,
    )
    return df.groupby("customer_id")["stock_code"].apply(list).to_dict()


def load_recs_llm(con: sqlite3.Connection, tier: str, run_id: str) -> dict[str, list[str]]:
    df = pd.read_sql(
        "SELECT customer_id, recommendations FROM llm_results WHERE tier=? AND run_id=?",
        con,
        params=(tier, run_id),
    )
    out: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        try:
            recs = json.loads(row["recommendations"])
        except Exception:
            recs = []
        out[row["customer_id"]] = [str(c).upper() for c in recs]
    return out


def evaluate_one(
    model: str,
    tier: str,
    recs_by_cust: dict[str, list[str]],
    holdout: pd.DataFrame,
    customers: pd.DataFrame,
    catalog_size: int,
    run_id: str,
) -> tuple[pd.DataFrame, dict]:
    truth = holdout.groupby("customer_id")["stock_code"].apply(set)
    rows = []
    for cid, truth_set in truth.items():
        ranked = recs_by_cust.get(cid, [])
        rows.append(
            {
                "customer_id": cid,
                "model": model,
                "tier": tier,
                "k": TOP_K,
                "hr_at_k": _hr(ranked, truth_set, TOP_K),
                "ndcg_at_k": _ndcg(ranked, truth_set, TOP_K),
                "run_id": run_id,
            }
        )
    df = pd.DataFrame(rows).merge(customers, on="customer_id", how="left")
    all_recs = {c for recs in recs_by_cust.values() for c in recs}
    cov = len(all_recs) / catalog_size if catalog_size else 0.0
    metrics = {
        "hr_mean": float(df["hr_at_k"].mean()) if len(df) else 0.0,
        "ndcg_mean": float(df["ndcg_at_k"].mean()) if len(df) else 0.0,
        "coverage": float(cov),
        "n": int(len(df)),
        "n_with_recs": int(sum(1 for v in recs_by_cust.values() if v)),
    }
    return df.drop(columns=["segment"], errors="ignore"), metrics


def write_eval(df: pd.DataFrame, con: sqlite3.Connection, model: str, tier: str) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_results (
            customer_id TEXT, model TEXT, tier TEXT, k INTEGER,
            hr_at_k REAL, ndcg_at_k REAL, freq_tier TEXT, run_id TEXT,
            PRIMARY KEY (customer_id, model, tier, run_id)
        )
        """
    )
    con.execute("DELETE FROM eval_results WHERE model=? AND tier=?", (model, tier))
    df.to_sql("eval_results", con, if_exists="append", index=False)
    con.commit()


def write_run_log(
    con: sqlite3.Connection, run_id: str, model: str, tier: str, metrics: dict, src_run_id: str
) -> None:
    params = {"top_k": TOP_K, "source_run_id": src_run_id, "n_with_recs": metrics["n_with_recs"]}
    con.execute(
        "INSERT INTO run_log (run_id, script, model, tier, params, n_customers, hr_mean, ndcg_mean, coverage, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            "evaluate.py",
            model,
            tier,
            json.dumps(params),
            metrics["n"],
            metrics["hr_mean"],
            metrics["ndcg_mean"],
            metrics["coverage"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    con.commit()


def latest_llm_run_id(con: sqlite3.Connection) -> str | None:
    row = con.execute(
        "SELECT run_id FROM run_log WHERE script='llm.py' ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-run-id", default=None, help="LLM run_id to evaluate (default: latest)")
    parser.add_argument("--skip-cf", action="store_true", help="Skip CF re-evaluation")
    args = parser.parse_args()

    con = sqlite3.connect(DB_PATH)
    try:
        holdout = pd.read_sql("SELECT customer_id, stock_code, freq_tier FROM holdout", con)
        customers = pd.read_sql("SELECT customer_id, freq_tier, segment FROM customers", con)
        catalog_size = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]

        llm_run_id = args.llm_run_id or latest_llm_run_id(con)
        if llm_run_id is None:
            print("ERROR: no llm.py runs found in run_log")
            return 1

        # Restrict to the customer set actually scored by the LLM run, so CF and LLM are compared on
        # the same customers (smoke runs only score a 50-customer stratified sample).
        scored = pd.read_sql(
            "SELECT DISTINCT customer_id FROM llm_results WHERE run_id=?", con, params=(llm_run_id,)
        )
        scored_set = set(scored["customer_id"])
        holdout = holdout[holdout["customer_id"].isin(scored_set)]
        print(f"evaluating on {holdout['customer_id'].nunique()} customers (LLM run_id={llm_run_id})")

        eval_run_id = str(uuid.uuid4())
        summary_rows = []

        if not args.skip_cf:
            cf_recs = {c: r for c, r in load_recs_cf(con).items() if c in scored_set}
            df, m = evaluate_one("cf", "baseline", cf_recs, holdout, customers, catalog_size, eval_run_id)
            write_eval(df, con, "cf", "baseline")
            write_run_log(con, eval_run_id, "cf", "baseline", m, src_run_id="phase5")
            summary_rows.append(("cf", "baseline", m))

        for tier in TIERS:
            recs = load_recs_llm(con, tier, llm_run_id)
            df, m = evaluate_one("llm", tier, recs, holdout, customers, catalog_size, eval_run_id)
            write_eval(df, con, "llm", tier)
            write_run_log(con, eval_run_id, "llm", tier, m, src_run_id=llm_run_id)
            summary_rows.append(("llm", tier, m))

        print("\n=== summary (k=10) ===")
        print(f"{'model':6s} {'tier':10s} {'HR':>8s} {'NDCG':>8s} {'cov':>8s} {'n':>6s} {'n_recs':>8s}")
        for model, tier, m in summary_rows:
            print(
                f"{model:6s} {tier:10s} {m['hr_mean']:>8.4f} {m['ndcg_mean']:>8.4f} "
                f"{m['coverage']:>8.4f} {m['n']:>6d} {m['n_with_recs']:>8d}"
            )

        print("\n=== HR@10 by freq_tier ===")
        eval_df = pd.read_sql(
            "SELECT model, tier, freq_tier, hr_at_k, ndcg_at_k FROM eval_results WHERE run_id=? OR (model='cf' AND tier='baseline')",
            con,
            params=(eval_run_id,),
        )
        pivot_hr = eval_df.pivot_table(
            index="freq_tier", columns=["model", "tier"], values="hr_at_k", aggfunc="mean"
        ).round(4)
        print(pivot_hr.to_string())

        print("\n=== NDCG@10 by freq_tier ===")
        pivot_ndcg = eval_df.pivot_table(
            index="freq_tier", columns=["model", "tier"], values="ndcg_at_k", aggfunc="mean"
        ).round(4)
        print(pivot_ndcg.to_string())

        print(f"\neval_run_id: {eval_run_id}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
