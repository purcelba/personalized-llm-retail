"""Build the customer-recs explorer widget.

Emits:
  - docs/customer_explorer.html          (standalone page)
  - docs/customer_explorer_fragment.html (CSS-scoped fragment for inlining into Markdown)

All CSS is scoped to #ce-root and all classes are prefixed `ce-` so the fragment
can be dropped into blog_post.md without colliding with the surrounding page.
"""
import sqlite3, json, random
from collections import defaultdict

RUN_ID = "0bc24f66-b37f-4161-a5b7-bac66c5ca9ba"
K = 10
HISTORY_TOP = 15
TIERS = ["cold", "sparse", "moderate", "rich", "champion"]
PER_TIER = {"cold": 15, "sparse": 35, "moderate": 25, "rich": 15, "champion": 10}
OUT_STANDALONE = "docs/customer_explorer.html"
OUT_FRAGMENT = "docs/customer_explorer_fragment.html"
SEED = 42

random.seed(SEED)
con = sqlite3.connect("data/retail.db")
cur = con.cursor()

descs = {sc: d for sc, d in cur.execute("SELECT stock_code, description FROM products")}

holdout = defaultdict(set)
for cid, sc in cur.execute("SELECT customer_id, stock_code FROM holdout"):
    holdout[str(cid)].add(str(sc).upper())

tier_of = {str(c): t for c, t in cur.execute("SELECT customer_id, freq_tier FROM customers")}
freq_of = {str(c): f for c, f in cur.execute("SELECT customer_id, frequency FROM customers")}

cf_recs = defaultdict(list)
for cid, sc, r in cur.execute(
    "SELECT customer_id, stock_code, rank FROM cf_recommendations ORDER BY customer_id, rank"):
    cf_recs[str(cid)].append(str(sc).upper())

def load_llm(tier):
    out = {}
    for cid, recs in cur.execute(
        "SELECT customer_id, recommendations FROM llm_results WHERE run_id=? AND tier=? AND parse_ok=1",
        (RUN_ID, tier)):
        try:
            lst = json.loads(recs) if isinstance(recs, str) else recs
        except Exception:
            lst = []
        out[str(cid)] = [str(x).upper() for x in lst][:K]
    return out

llm_base = load_llm("llm_base")
llm_cf_recs = load_llm("llm_cf")

def history(cid):
    rows = cur.execute("""
        SELECT t.stock_code, SUM(t.quantity) AS units
        FROM transactions t
        WHERE t.customer_id=?
          AND t.invoice_id NOT IN (SELECT invoice_id FROM holdout WHERE customer_id=?)
        GROUP BY t.stock_code
        ORDER BY units DESC
        LIMIT ?""", (cid, cid, HISTORY_TOP)).fetchall()
    return [{"code": sc, "desc": descs.get(sc, ""), "units": int(u)} for sc, u in rows]

candidates = [c for c in holdout
              if c in cf_recs and c in llm_base and c in llm_cf_recs and c in tier_of]

def hits(recs, gt): return sum(1 for r in recs if r in gt)

scored = []
for c in candidates:
    gt = holdout[c]
    s = hits(cf_recs[c], gt) + hits(llm_base[c], gt) + hits(llm_cf_recs[c], gt)
    scored.append((c, s))

by_tier = defaultdict(list)
for c, s in scored:
    by_tier[tier_of[c]].append((c, s))

sample = []
for t in TIERS:
    pool = by_tier[t]
    if not pool: continue
    hits_pool = [c for c, s in pool if s > 0]
    miss_pool = [c for c, s in pool if s == 0]
    n = PER_TIER[t]
    n_hits = min(int(n * 0.75), len(hits_pool))
    n_miss = min(n - n_hits, len(miss_pool))
    random.shuffle(hits_pool); random.shuffle(miss_pool)
    sample.extend(hits_pool[:n_hits] + miss_pool[:n_miss])

random.shuffle(sample)

def rec_with_desc(codes, gt):
    return [{"code": c, "desc": descs.get(c, ""), "hit": c in gt} for c in codes]

data = []
for cid in sample:
    gt = holdout[cid]
    data.append({
        "id": cid,
        "tier": tier_of[cid],
        "freq": freq_of.get(cid),
        "history": history(cid),
        "truth": [{"code": c, "desc": descs.get(c, "")} for c in sorted(gt)],
        "cf": rec_with_desc(cf_recs[cid], gt),
        "lb": rec_with_desc(llm_base[cid], gt),
        "lc": rec_with_desc(llm_cf_recs[cid], gt),
    })

print(f"Sampled {len(data)} customers across tiers:",
      {t: sum(1 for d in data if d['tier']==t) for t in TIERS})

payload = json.dumps(data, separators=(",", ":"))

# All CSS scoped under #ce-root. All classes prefixed ce-.
STYLE = """<style>
  #ce-root {
    --ce-cf:#3b6bb2; --ce-lb:#dd7a3d; --ce-lc:#3f9c66;
    --ce-bg:#f4f5f8; --ce-card:#ffffff; --ce-border:#e6e8ec;
    --ce-muted:#6b7280; --ce-text:#0f172a; --ce-heading:#0b1220;
    --ce-hit-bg:#dcf5e0; --ce-hit-border:#3f9c66;
    --ce-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 12px rgba(15,23,42,0.04);
    --ce-shadow-lg: 0 2px 4px rgba(15,23,42,0.06), 0 8px 24px rgba(15,23,42,0.06);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    font-feature-settings: 'cv11', 'ss01';
    font-size:14px; color:var(--ce-text); background:var(--ce-bg);
    -webkit-font-smoothing: antialiased;
    letter-spacing:-0.005em;
    padding:22px; border-radius:12px; border:1px solid var(--ce-border);
    margin: 24px 0;
  }
  #ce-root *, #ce-root *::before, #ce-root *::after { box-sizing: border-box; }
  #ce-root .ce-toolbar {
    display:flex; gap:18px; align-items:center; flex-wrap:wrap;
    background:var(--ce-card); border:1px solid var(--ce-border); border-radius:12px;
    padding:14px 18px; margin-bottom:16px;
    box-shadow: var(--ce-shadow);
  }
  #ce-root .ce-label {
    font-weight:600; color:var(--ce-heading); font-size:11px;
    text-transform:uppercase; letter-spacing:0.08em;
    display:flex; align-items:center; gap:10px;
  }
  #ce-root select {
    padding:9px 32px 9px 12px; border:1px solid var(--ce-border); border-radius:8px;
    background:white; font-size:14px; color:var(--ce-text); min-width:280px;
    font-family:inherit; font-weight:500;
    appearance:none;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='none' stroke='%236b7280' stroke-width='1.6' d='M3 4.5l3 3 3-3'/%3E%3C/svg%3E");
    background-repeat:no-repeat; background-position:right 10px center;
    cursor:pointer; transition: border-color .15s, box-shadow .15s;
  }
  #ce-root select:hover { border-color:#c7cad1; }
  #ce-root select:focus { outline:none; border-color:#7aa7e6; box-shadow:0 0 0 3px rgba(122,167,230,0.25); }
  #ce-root .ce-meta { margin-left:auto; color:var(--ce-muted); font-size:13px; display:flex; gap:6px; flex-wrap:wrap; }
  #ce-root .ce-pill {
    display:inline-flex; align-items:center; padding:4px 11px; border-radius:999px;
    background:#eef1f6; color:#334155; font-weight:600; font-size:12px;
  }
  #ce-root .ce-grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
  @media (max-width: 900px) { #ce-root .ce-grid-3 { grid-template-columns:1fr; } }
  #ce-root .ce-card {
    background:var(--ce-card); border:1px solid var(--ce-border); border-radius:12px;
    padding:16px 18px; box-shadow: var(--ce-shadow);
    transition: box-shadow .2s;
    position:relative;
  }
  #ce-root .ce-card:hover { box-shadow: var(--ce-shadow-lg); }
  #ce-root .ce-card-title {
    margin-bottom:12px;
    display:flex; align-items:baseline; justify-content:space-between; gap:10px;
  }
  #ce-root .ce-title {
    font-weight:800; font-size:15px; color:var(--ce-heading);
    letter-spacing:-0.015em;
  }
  #ce-root .ce-sub {
    font-weight:500; color:var(--ce-muted); font-size:11px;
    text-transform:uppercase; letter-spacing:0.06em;
  }
  #ce-root .ce-accent::before {
    content:""; position:absolute; left:18px; right:18px; top:0; height:3px;
    border-radius:0 0 3px 3px; background:var(--ce-cf);
  }
  #ce-root .ce-accent.ce-m-cf::before { background:var(--ce-cf); }
  #ce-root .ce-accent.ce-m-lb::before { background:var(--ce-lb); }
  #ce-root .ce-accent.ce-m-lc::before { background:var(--ce-lc); }
  #ce-root .ce-accent.ce-m-cf .ce-title { color:var(--ce-cf); }
  #ce-root .ce-accent.ce-m-lb .ce-title { color:var(--ce-lb); }
  #ce-root .ce-accent.ce-m-lc .ce-title { color:var(--ce-lc); }
  #ce-root .ce-recs, #ce-root .ce-items {
    margin:0; padding:0; list-style:none;
    max-height: 192px; overflow-y:auto;
    scrollbar-width: thin; scrollbar-color: #cbd5e1 transparent;
  }
  #ce-root .ce-recs::-webkit-scrollbar, #ce-root .ce-items::-webkit-scrollbar { width:8px; }
  #ce-root .ce-recs::-webkit-scrollbar-thumb, #ce-root .ce-items::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:4px; }
  #ce-root .ce-recs li, #ce-root .ce-items li {
    display:flex; gap:10px; align-items:baseline;
    padding:6px 8px; border-radius:6px; line-height:1.45;
    font-size:13px; color:#1f2937;
  }
  #ce-root .ce-recs li + li, #ce-root .ce-items li + li { margin-top:1px; }
  #ce-root .ce-num {
    color:#9ca3af; width:20px; text-align:right; flex-shrink:0;
    font-variant-numeric:tabular-nums; font-weight:600; font-size:12px;
  }
  #ce-root .ce-units {
    color:var(--ce-muted); margin-left:auto;
    font-variant-numeric:tabular-nums; font-size:11px; font-weight:600;
    background:#f1f3f5; padding:2px 7px; border-radius:999px;
  }
  #ce-root code {
    background:#f1f3f5; padding:2px 7px; border-radius:4px;
    font-family: 'JetBrains Mono', ui-monospace, "SF Mono", Menlo, monospace;
    font-size:11.5px; font-weight:500; color:#1f2937; letter-spacing:-0.01em;
  }
  #ce-root .ce-hit {
    background:var(--ce-hit-bg);
    box-shadow: inset 3px 0 0 var(--ce-hit-border);
    font-weight:500;
  }
  #ce-root .ce-hit code { background:#bfe8c7; color:#0f3a1c; }
  #ce-root .ce-legend {
    color:var(--ce-muted); font-size:12px; margin-top:14px;
    display:flex; gap:14px; flex-wrap:wrap; align-items:center;
  }
  #ce-root .ce-swatch {
    display:inline-block; width:12px; height:12px; border-radius:3px;
    margin-right:6px; vertical-align:middle;
    background:var(--ce-hit-bg); border:1px solid rgba(0,0,0,0.06);
    box-shadow: inset 3px 0 0 var(--ce-hit-border);
  }
</style>"""

FONTS = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">"""

WIDGET = """<div id="ce-root">
  <div class="ce-toolbar">
    <label class="ce-label">Customer
      <select id="ce-cust"></select>
    </label>
    <label class="ce-label">Model
      <select id="ce-model" style="min-width:220px;">
        <option value="cf">CF baseline (ALS)</option>
        <option value="lb">LLM only</option>
        <option value="lc" selected>LLM + CF</option>
      </select>
    </label>
    <div class="ce-meta" id="ce-meta"></div>
  </div>
  <div class="ce-grid-3">
    <div class="ce-card" id="ce-hist"></div>
    <div class="ce-card" id="ce-truth"></div>
    <div class="ce-card ce-accent" id="ce-preds"></div>
  </div>
  <div class="ce-legend">
    <span><span class="ce-swatch"></span>hit — recommended product appears in the held-out invoice</span>
    <span style="margin-left:auto;">Sample of __N__ customers across all frequency tiers.</span>
  </div>
</div>
<script>
(function(){
  const DATA = __PAYLOAD__;
  const custSel = document.getElementById('ce-cust');
  const modelSel = document.getElementById('ce-model');
  const meta = document.getElementById('ce-meta');
  const els = {
    hist: document.getElementById('ce-hist'),
    truth: document.getElementById('ce-truth'),
    preds: document.getElementById('ce-preds'),
  };
  const MODEL_LABEL = {cf:'CF baseline (ALS)', lb:'LLM only', lc:'LLM + CF'};
  const tierLabel = {cold:'cold (1–2)', sparse:'sparse (3–5)', moderate:'moderate (6–15)', rich:'rich (16–30)', champion:'champion (31+)'};
  const tierOrder = ['cold','sparse','moderate','rich','champion'];
  const sorted = DATA.slice().sort((a,b)=>{
    const ta = tierOrder.indexOf(a.tier), tb = tierOrder.indexOf(b.tier);
    if (ta !== tb) return ta - tb;
    const ha = a.cf.filter(x=>x.hit).length + a.lb.filter(x=>x.hit).length + a.lc.filter(x=>x.hit).length;
    const hb = b.cf.filter(x=>x.hit).length + b.lb.filter(x=>x.hit).length + b.lc.filter(x=>x.hit).length;
    if (ha !== hb) return hb - ha;
    return a.id.localeCompare(b.id);
  });
  function rebuildCustList(){
    const m = modelSel.value;
    const prev = custSel.value;
    custSel.innerHTML = '';
    sorted.forEach(d => {
      const o = document.createElement('option');
      o.value = d.id;
      const h = d[m].filter(x=>x.hit).length;
      o.textContent = `Customer #${d.id}  •  ${tierLabel[d.tier]}  •  ${h} hit${h===1?'':'s'}`;
      custSel.appendChild(o);
    });
    if (prev && sorted.some(d=>d.id===prev)) custSel.value = prev;
  }
  function esc(s){ return s==null ? '' : String(s).replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
  function recHTML(recs){
    return recs.map((r,i)=>`<li class="${r.hit?'ce-hit':''}"><span class="ce-num">${i+1}.</span><code>${esc(r.code)}</code><span>${esc(r.desc)}</span></li>`).join('');
  }
  function itemHTML(items, showUnits){
    return items.map(r => {
      const u = showUnits ? `<span class="ce-units">${r.units}u</span>` : '';
      return `<li><code>${esc(r.code)}</code><span>${esc(r.desc)}</span>${u}</li>`;
    }).join('');
  }
  function cardHead(title, sub){
    return `<div class="ce-card-title"><span class="ce-title">${title}</span>${sub?`<span class="ce-sub">${sub}</span>`:''}</div>`;
  }
  function render(){
    const d = DATA.find(x => x.id === custSel.value);
    if (!d) return;
    const m = modelSel.value;
    meta.innerHTML = `<span class="ce-pill">${tierLabel[d.tier]}</span><span class="ce-pill">${d.freq} prior invoices</span>`;
    els.hist.innerHTML = cardHead('Purchase history', `top ${d.history.length} by units`) + `<ul class="ce-items">${itemHTML(d.history, true)}</ul>`;
    els.truth.innerHTML = cardHead('Next purchases (held-out)', `${d.truth.length} items`) + `<ul class="ce-items">${itemHTML(d.truth, false)}</ul>`;
    const recs = d[m];
    const h = recs.filter(r=>r.hit).length;
    els.preds.className = `ce-card ce-accent ce-m-${m}`;
    els.preds.innerHTML = cardHead(`${MODEL_LABEL[m]} predictions`, `${h} hit${h===1?'':'s'} in top 10`) + `<ol class="ce-recs">${recHTML(recs)}</ol>`;
  }
  custSel.addEventListener('change', render);
  modelSel.addEventListener('change', ()=>{ rebuildCustList(); render(); });
  rebuildCustList();
  custSel.value = sorted[0].id;
  render();
})();
</script>"""

WIDGET = WIDGET.replace("__PAYLOAD__", payload).replace("__N__", str(len(data)))

STANDALONE = f"""<!doctype html><html><head><meta charset="utf-8"><title>Customer recs explorer</title>
{FONTS}
{STYLE}
</head><body style="margin:0;background:#f4f5f8;padding:8px;">
{WIDGET}
</body></html>
"""

FRAGMENT = f"{FONTS}\n{STYLE}\n{WIDGET}\n"

with open(OUT_STANDALONE, "w") as f:
    f.write(STANDALONE)
print(f"Wrote {OUT_STANDALONE} ({len(STANDALONE):,} chars)")

with open(OUT_FRAGMENT, "w") as f:
    f.write(FRAGMENT)
print(f"Wrote {OUT_FRAGMENT} ({len(FRAGMENT):,} chars)")
