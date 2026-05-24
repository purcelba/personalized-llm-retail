"""
Phase 8 — Reporting
Aggregate eval_results and produce comparison tables and plots.
Saves plots to data/plots/.
"""

from __future__ import annotations

import os
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DB_PATH = "data/retail.db"
PLOTS_DIR = "data/plots"
MODEL_TIER_ORDER = [("cf", "baseline"), ("llm", "llm_base"), ("llm", "llm_cf")]


def load_eval(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT e.customer_id, e.model, e.tier, e.hr_at_k, e.ndcg_at_k, e.freq_tier,
               c.segment
        FROM eval_results e
        LEFT JOIN customers c USING (customer_id)
        """,
        con,
    )


def load_run_summary(con: sqlite3.Connection) -> pd.DataFrame:
    eval_runs = pd.read_sql(
        """
        SELECT model, tier, hr_mean, ndcg_mean, coverage, n_customers, ts
        FROM run_log
        WHERE script='evaluate.py'
        ORDER BY ts DESC
        """,
        con,
    )
    # Latest row per (model, tier)
    return eval_runs.drop_duplicates(subset=["model", "tier"]).reset_index(drop=True)


def _label(row) -> str:
    return "CF" if row["model"] == "cf" else row["tier"]


def print_summary(summary: pd.DataFrame) -> None:
    summary = summary.copy()
    summary["label"] = summary.apply(_label, axis=1)
    summary = summary.set_index("label").reindex([_label({"model": m, "tier": t}) for m, t in MODEL_TIER_ORDER])
    print("\n=== Overall metrics (k=10) ===")
    print(summary[["hr_mean", "ndcg_mean", "coverage", "n_customers"]].round(4).to_string())


def print_pivots(eval_df: pd.DataFrame) -> None:
    for metric in ("hr_at_k", "ndcg_at_k"):
        pv = eval_df.pivot_table(
            index="freq_tier", columns=["model", "tier"], values=metric, aggfunc="mean"
        ).round(4)
        pv = pv.reindex(columns=MODEL_TIER_ORDER)
        print(f"\n=== {metric} by freq_tier ===")
        print(pv.to_string())


def plot_hr_by_freq_tier(eval_df: pd.DataFrame, path: str) -> None:
    pv = eval_df.pivot_table(
        index="freq_tier", columns=["model", "tier"], values="hr_at_k", aggfunc="mean"
    )
    pv = pv.reindex(columns=MODEL_TIER_ORDER)
    tier_order = ["cold", "sparse", "moderate", "rich", "champion"]
    pv = pv.reindex(index=[t for t in tier_order if t in pv.index])

    fig, ax = plt.subplots(figsize=(10, 5))
    n_cols = len(pv.columns)
    width = 0.8 / n_cols
    x = np.arange(len(pv.index))
    colors = ["#666"] + plt.cm.viridis(np.linspace(0.15, 0.85, n_cols - 1)).tolist()
    for i, (mod, tier) in enumerate(pv.columns):
        label = "CF baseline" if mod == "cf" else tier
        ax.bar(x + i * width - 0.4 + width / 2, pv[(mod, tier)].values, width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(pv.index)
    ax.set_ylabel("HR@10")
    ax.set_xlabel("Frequency tier")
    ax.set_title("HR@10 by frequency tier — CF vs LLM context tiers")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def plot_heatmap_segment_tier(eval_df: pd.DataFrame, path: str) -> None:
    df = eval_df.dropna(subset=["segment"]).copy()
    df["col"] = df.apply(_label, axis=1)
    col_order = [_label({"model": m, "tier": t}) for m, t in MODEL_TIER_ORDER]
    seg_order = ["champions", "loyal", "at_risk", "hibernating"]
    pv = df.pivot_table(index="segment", columns="col", values="hr_at_k", aggfunc="mean")
    pv = pv.reindex(index=[s for s in seg_order if s in pv.index], columns=col_order)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pv.columns)))
    ax.set_xticklabels(pv.columns)
    ax.set_yticks(np.arange(len(pv.index)))
    ax.set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.15 else "black", fontsize=9)
    ax.set_title("HR@10 by RFM segment × model/tier")
    fig.colorbar(im, ax=ax, label="HR@10")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def top_findings(eval_df: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    summary = summary.copy()
    summary["label"] = summary.apply(_label, axis=1)
    cf = summary[summary["model"] == "cf"].iloc[0]
    llm = summary[summary["model"] == "llm"]
    best = llm.loc[llm["hr_mean"].idxmax()]
    findings.append(
        f"Overall HR@10 winner: {('CF baseline' if best['hr_mean'] <= cf['hr_mean'] else 'LLM ' + best['tier'])} "
        f"(CF={cf['hr_mean']:.3f}, best LLM={best['tier']} at {best['hr_mean']:.3f})"
    )
    # Lift on cold
    pv = eval_df.pivot_table(index="freq_tier", columns=["model", "tier"], values="hr_at_k", aggfunc="mean")
    pv = pv.reindex(columns=MODEL_TIER_ORDER)
    if "cold" in pv.index:
        cf_cold = pv.loc["cold", ("cf", "baseline")]
        best_llm_cold = pv.loc["cold", [c for c in pv.columns if c[0] == "llm"]].max()
        findings.append(
            f"Cold tier — CF HR@10={cf_cold:.3f}, best LLM={best_llm_cold:.3f} "
            f"({'LLM lift' if best_llm_cold > cf_cold else 'no LLM lift'})"
        )
    if "moderate" in pv.index:
        cf_mod = pv.loc["moderate", ("cf", "baseline")]
        best_llm_mod = pv.loc["moderate", [c for c in pv.columns if c[0] == "llm"]].max()
        findings.append(
            f"Moderate tier — CF HR@10={cf_mod:.3f}, best LLM={best_llm_mod:.3f} "
            f"({'LLM lift' if best_llm_mod > cf_mod else 'no LLM lift'})"
        )
    # Coverage
    cov = summary.set_index("label")["coverage"].reindex([_label({"model": m, "tier": t}) for m, t in MODEL_TIER_ORDER])
    findings.append("Coverage (catalog %): " + ", ".join(f"{lbl}={v:.1%}" for lbl, v in cov.items()))
    return findings


def main() -> int:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        eval_df = load_eval(con)
        summary = load_run_summary(con)
        if eval_df.empty:
            print("ERROR: eval_results empty — run src.evaluate first")
            return 1
        print_summary(summary)
        print_pivots(eval_df)
        plot_hr_by_freq_tier(eval_df, os.path.join(PLOTS_DIR, "hr_by_tier.png"))
        plot_heatmap_segment_tier(eval_df, os.path.join(PLOTS_DIR, "heatmap_segment_tier.png"))
        print("\n=== Top findings ===")
        for f in top_findings(eval_df, summary):
            print(f"  • {f}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
