"""
Phase 6 — LLM Runs (llm_base, llm_cf)
Two groups: llm_base (history only) vs llm_cf (history + CF neighbors).
Assembles tier-appropriate prompts and calls the Claude API.
Checkpoints every N calls. Writes to the llm_results table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

DB_PATH = "data/retail.db"
RANDOM_SEED = 42
TOP_K = 10
TEMPERATURE = 0
MAX_TOKENS = 1200
CHECKPOINT_EVERY = 50
SAMPLE_SIZE_SMOKE = 50
MODEL_ID_FULL = "claude-sonnet-4-6"
MODEL_ID_DEV = "claude-haiku-4-5-20251001"
TIERS = ["llm_base", "llm_cf"]
N_CANDIDATES = 50

SYSTEM_PROMPT = (
    "You are a product recommendation assistant for an online retailer. "
    f"Recommend exactly {TOP_K} products the customer is likely to buy next. "
    "The user message contains a CANDIDATE LIST formatted as `- description (STOCKCODE)`. "
    "Strongly prefer picking from that list, but you may pick other catalog products if you "
    "have a clear reason. Do NOT recommend products from the customer's purchase history "
    "(those are already owned). "
    "CRITICAL: each entry in the `recommendations` array must be ONLY the bare stock_code "
    "string (e.g. \"22417\" or \"85123A\") — do NOT include the description, parentheses, "
    "or any other text. "
    "Respond with ONLY a JSON object of the form: "
    '{"recommendations": ["STOCKCODE1", ...], "rationale": "..."} '
    "Order recommendations best to worst."
)


# ---------- context loaders ----------


def load_purchase_history(con: sqlite3.Connection, customer_id: str) -> pd.DataFrame:
    holdout_invs = pd.read_sql(
        "SELECT invoice_id FROM holdout WHERE customer_id = ?",
        con,
        params=(customer_id,),
    )["invoice_id"].tolist()
    placeholders = ",".join("?" * len(holdout_invs)) if holdout_invs else "''"
    sql = f"""
        SELECT t.stock_code, p.description, SUM(t.quantity) AS units
        FROM transactions t
        LEFT JOIN products p ON p.stock_code = t.stock_code
        WHERE t.customer_id = ?
          AND t.invoice_id NOT IN ({placeholders})
        GROUP BY t.stock_code
        ORDER BY units DESC
    """
    params = [customer_id] + holdout_invs
    return pd.read_sql(sql, con, params=params)


def get_cf_neighbors(con: sqlite3.Connection, customer_id: str, n: int = 10) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT cr.stock_code, p.description
        FROM cf_recommendations cr
        LEFT JOIN products p ON p.stock_code = cr.stock_code
        WHERE cr.customer_id = ?
        ORDER BY cr.rank ASC
        LIMIT ?
        """,
        con,
        params=(customer_id, n),
    )


def build_candidate_pool(
    con: sqlite3.Connection,
    customer_id: str,
    tier: str,
    purchased: set[str],
) -> pd.DataFrame:
    pool: list[pd.DataFrame] = []
    # Pull a wide popular pool (top 200) so even champions have enough novel options.
    pool.append(
        pd.read_sql(
            "SELECT stock_code, description FROM products ORDER BY popularity_rank ASC LIMIT 200",
            con,
        )
    )
    if tier == "llm_cf":
        pool.append(get_cf_neighbors(con, customer_id, n=TOP_K))
    cands = pd.concat(pool, ignore_index=True).drop_duplicates(subset=["stock_code"])
    cands = cands[~cands["stock_code"].isin(purchased)].head(N_CANDIDATES)
    return cands.reset_index(drop=True)


# ---------- prompt assembly ----------


def fmt_history(history: pd.DataFrame, max_rows: int = 25) -> str:
    rows = history.head(max_rows)
    lines = [f"- {r.description} ({r.stock_code}, purchased {int(r.units)})" for r in rows.itertuples()]
    if len(history) > max_rows:
        lines.append(f"... and {len(history) - max_rows} more products")
    return "\n".join(lines) if lines else "(no prior purchases)"


def fmt_products(df: pd.DataFrame) -> str:
    return "\n".join(f"- {r.description} ({r.stock_code})" for r in df.itertuples())


def build_prompt(
    customer_id: str,
    tier: str,
    con: sqlite3.Connection,
    all_codes: set[str],
) -> tuple[str, set[str]]:
    history = load_purchase_history(con, customer_id)
    purchased = set(history["stock_code"])
    candidates = build_candidate_pool(con, customer_id, tier, purchased)
    valid_codes = (all_codes - purchased)

    blocks = [f"Customer purchase history:\n{fmt_history(history)}"]

    if tier == "llm_cf":
        cf = get_cf_neighbors(con, customer_id, n=TOP_K)
        if len(cf):
            blocks.append(
                "Customers with similar purchase patterns also bought:\n" + fmt_products(cf)
            )

    blocks.append(
        f"CANDIDATE LIST (suggested — strongly prefer these, but you may pick other catalog products if you have a clear reason):\n"
        + fmt_products(candidates)
    )
    prompt = "\n\n".join(blocks)
    return prompt, valid_codes


# ---------- API call + parse ----------


class RateLimiter:
    """Min-interval gate, thread-safe. min_interval = 60 / target_rpm."""

    def __init__(self, target_rpm: float) -> None:
        self.min_interval = 60.0 / target_rpm if target_rpm > 0 else 0.0
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            delta = now - self.last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self.last = time.monotonic()


def call_api(client, prompt: str, model_id: str, rate_limiter: RateLimiter | None = None) -> str:
    delay = 2.0
    last_err = None
    for _ in range(8):
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            last_err = e
            etype = type(e).__name__
            msg = str(e)
            is_rl = etype == "RateLimitError" or "rate_limit" in msg or "429" in msg
            if is_rl or etype in {"APIConnectionError", "APITimeoutError", "InternalServerError"}:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            raise
    raise last_err


def _extract_json_object(text: str) -> str | None:
    """Find the first balanced {...} block, respecting strings and escapes."""
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        # No close found from this start; try next "{"
        start = text.find("{", start + 1)
    return None


def parse_response(raw: str, valid_codes: set[str]) -> tuple[list[str], str]:
    text = raw.strip()
    obj = None
    # 1. Direct parse
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Balanced-brace extraction (handles prose preambles, ```json fences, trailing text)
    if obj is None:
        candidate = _extract_json_object(text)
        if candidate is not None:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                pass
    # 3. Fallback: rationale was truncated by max_tokens — grab the recommendations array
    if obj is None:
        m = re.search(r'"recommendations"\s*:\s*(\[[^\]]*\])', text, re.DOTALL)
        if m:
            try:
                obj = {"recommendations": json.loads(m.group(1)), "rationale": ""}
            except json.JSONDecodeError:
                pass
    if obj is None:
        raise json.JSONDecodeError("could not extract JSON from response", text, 0)
    raw_recs = obj.get("recommendations", [])
    rationale = obj.get("rationale", "")
    filtered: list[str] = []
    seen = set()
    for code in raw_recs:
        s = str(code).strip()
        c = s.upper()
        if c not in valid_codes:
            # Fallback: extract trailing "(CODE)" if model echoed "description (CODE)"
            m = re.search(r"\(([A-Z0-9]+)\)\s*$", c)
            if m and m.group(1) in valid_codes:
                c = m.group(1)
        if c in valid_codes and c not in seen:
            filtered.append(c)
            seen.add(c)
        if len(filtered) >= TOP_K:
            break
    return filtered, rationale


# ---------- db setup + checkpoint ----------


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_results (
            customer_id     TEXT,
            tier            TEXT,
            run_id          TEXT,
            model           TEXT,
            recommendations TEXT,
            rationale       TEXT,
            parse_ok        INTEGER,
            ts              TEXT,
            PRIMARY KEY (customer_id, tier, run_id)
        )
        """
    )
    con.commit()


def already_done(con: sqlite3.Connection, tier: str, run_id: str) -> set[str]:
    return set(
        pd.read_sql(
            "SELECT customer_id FROM llm_results WHERE tier=? AND run_id=?",
            con,
            params=(tier, run_id),
        )["customer_id"]
    )


def insert_result(
    con: sqlite3.Connection,
    customer_id: str,
    tier: str,
    run_id: str,
    model: str,
    recs: list[str],
    rationale: str,
    parse_ok: bool,
) -> None:
    con.execute(
        "INSERT OR REPLACE INTO llm_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            customer_id,
            tier,
            run_id,
            model,
            json.dumps(recs),
            rationale,
            int(parse_ok),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    con.commit()


# ---------- run_log ----------


def write_run_log(
    con: sqlite3.Connection,
    run_id: str,
    tier: str,
    model: str,
    n_customers: int,
    parse_failures: int,
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
        "model_id": model,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_k": TOP_K,
        "n_candidates": N_CANDIDATES,
        "parse_failures": parse_failures,
    }
    con.execute(
        "INSERT INTO run_log (run_id, script, model, tier, params, n_customers, hr_mean, ndcg_mean, coverage, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            "llm.py",
            "llm",
            tier,
            json.dumps(params),
            n_customers,
            None,
            None,
            None,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    con.commit()


# ---------- sampling ----------


def stratified_sample(
    con: sqlite3.Connection, n: int, exclude: set[str] | None = None
) -> list[str]:
    exclude = exclude or set()
    holdout = pd.read_sql(
        """
        SELECT DISTINCT h.customer_id, h.freq_tier
        FROM holdout h
        """,
        con,
    )
    holdout = holdout[~holdout["customer_id"].isin(exclude)]
    rng = np.random.default_rng(RANDOM_SEED)
    tiers = holdout["freq_tier"].dropna().unique().tolist()
    per_tier = max(1, n // len(tiers))
    chosen: list[str] = []
    for t in sorted(tiers):
        sub = holdout[holdout["freq_tier"] == t]["customer_id"].tolist()
        if len(sub) <= per_tier:
            chosen.extend(sub)
        else:
            idx = rng.choice(len(sub), size=per_tier, replace=False)
            chosen.extend([sub[i] for i in idx])
    return chosen[:n]


def existing_customers(con: sqlite3.Connection, run_id: str) -> list[str]:
    return pd.read_sql(
        "SELECT DISTINCT customer_id FROM llm_results WHERE run_id=?",
        con,
        params=(run_id,),
    )["customer_id"].tolist()


# ---------- main ----------


def run_tier(
    tier: str,
    customers: list[str],
    con: sqlite3.Connection,
    run_id: str,
    model_id: str,
    client,
    workers: int,
    rate_limiter: RateLimiter,
) -> tuple[int, int]:
    done = already_done(con, tier, run_id)
    remaining = [c for c in customers if c not in done]
    print(f"\n[{tier}] processing {len(remaining)} of {len(customers)} customers (workers={workers})")
    if not remaining:
        return 0, 0

    all_codes = set(
        pd.read_sql("SELECT stock_code FROM products", con)["stock_code"].astype(str).str.upper()
    )

    t_prompt_start = time.perf_counter()
    prompts: dict[str, tuple[str, set[str]]] = {}
    for cid in remaining:
        prompts[cid] = build_prompt(cid, tier, con, all_codes)
    t_prompt = time.perf_counter() - t_prompt_start

    parse_failures = 0
    empty_recs = 0
    t_api_total = 0.0
    completed = 0
    tier_start = time.perf_counter()

    def _api(cid: str) -> tuple[str, str, set[str], float, Exception | None]:
        prompt, valid = prompts[cid]
        a0 = time.perf_counter()
        try:
            raw = call_api(client, prompt, model_id, rate_limiter)
            return cid, raw, valid, time.perf_counter() - a0, None
        except Exception as e:
            return cid, "", valid, time.perf_counter() - a0, e

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_api, cid) for cid in remaining]
        for fut in as_completed(futures):
            cid, raw, valid, dt, err = fut.result()
            t_api_total += dt
            completed += 1
            if err is not None:
                parse_failures += 1
                print(f"  api error customer={cid}: {err}")
                insert_result(con, cid, tier, run_id, model_id, [], "", False)
            else:
                try:
                    recs, rationale = parse_response(raw, valid)
                    insert_result(con, cid, tier, run_id, model_id, recs, rationale, True)
                    if not recs:
                        empty_recs += 1
                except Exception as parse_err:
                    parse_failures += 1
                    print(f"  parse fail customer={cid}: {parse_err}; raw[:120]={raw[:120]!r}")
                    insert_result(con, cid, tier, run_id, model_id, [], raw[:2000], False)
            if completed % CHECKPOINT_EVERY == 0:
                print(f"  [{tier}] {completed}/{len(remaining)} done (parse_failures={parse_failures}, empty={empty_recs})")

    elapsed = time.perf_counter() - tier_start
    print(
        f"[{tier}] complete in {elapsed:.1f}s; parse_failures={parse_failures}, empty_recs={empty_recs}\n"
        f"  prompt build: {t_prompt:.1f}s total ({t_prompt/len(remaining):.3f}s/customer)\n"
        f"  api: sum-of-call-time={t_api_total:.1f}s, wall={elapsed:.1f}s, "
        f"effective parallelism={t_api_total/elapsed:.1f}x"
    )
    return len(remaining), parse_failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run on stratified sample with dev model")
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE_SMOKE, help="smoke sample size")
    parser.add_argument("--run-id", default=None, help="resume an existing run_id")
    parser.add_argument("--tiers", default=",".join(TIERS), help="comma-separated tier list")
    parser.add_argument("--workers", type=int, default=4, help="concurrent API workers")
    parser.add_argument("--rpm", type=float, default=45.0, help="target requests per minute (under tier rate limit)")
    parser.add_argument("--extend", action="store_true", help="with --run-id and --n, keep existing customers and top up to --n by stratified sampling")
    parser.add_argument("--model", default=None, help="override model id (otherwise dev model if --smoke else full model)")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set (check .env)")
        return 1

    import anthropic

    client = anthropic.Anthropic()

    model_id = args.model or (MODEL_ID_DEV if args.smoke else MODEL_ID_FULL)
    run_id = args.run_id or str(uuid.uuid4())
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    print(f"run_id: {run_id} | model: {model_id} | smoke={args.smoke} | tiers={tiers}")

    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con)
        if args.extend:
            if not args.run_id:
                print("ERROR: --extend requires --run-id")
                return 1
            existing = existing_customers(con, args.run_id)
            need = max(0, args.n - len(existing))
            new = stratified_sample(con, need, exclude=set(existing)) if need else []
            customers = existing + new
            print(f"extend: keeping {len(existing)} existing, adding {len(new)} new (target n={args.n})")
        elif args.smoke:
            customers = stratified_sample(con, args.n)
        else:
            customers = pd.read_sql(
                "SELECT DISTINCT customer_id FROM holdout", con
            )["customer_id"].tolist()
        print(f"customer count: {len(customers)}")

        rate_limiter = RateLimiter(args.rpm)
        for tier in tiers:
            n, fails = run_tier(tier, customers, con, run_id, model_id, client, args.workers, rate_limiter)
            if n > 0:
                write_run_log(con, run_id, tier, model_id, n, fails)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
