# Example prompts — customer 12421 (moderate / loyal)

Real prompts emitted by `src/llm.py` for one customer across both LLM groups. The system prompt (shared across groups, omitted here) is in `src/llm.py::SYSTEM_PROMPT`. The candidate list is identical across groups in this example because the customer's purchase set didn't displace any of the top-200 popular items into the CF tail — for customers with more overlap, candidates can differ between groups.

**Customer features:** `freq_tier=moderate, segment=loyal, recency=15d, frequency=6, monetary=£1,098`

**Ground truth held-out invoice (never shown to model):** `[21108, 21907, 22307, 23234, 23267, 23311, 23382, 47567B]`

---

## `llm_base` — purchase history only (~3,404 chars)

```
Customer purchase history:
- LOVE HEART TRINKET POT (22066, purchased 60)
- GIN & TONIC DIET GREETING CARD  (21519, purchased 60)
- STRAWBERRY CERAMIC TRINKET BOX (21232, purchased 60)
- CERAMIC STRAWBERRY CAKE MONEY BANK (22646, purchased 36)
- 60 TEATIME FAIRY CAKE CASES (84991, purchased 24)
- MINI CAKE STAND T-LIGHT HOLDER (22893, purchased 24)
- CARD CHRISTMAS VILLAGE (22818, purchased 24)
- BANQUET BIRTHDAY  CARD   (22026, purchased 24)
- BOOZE & WOMEN GREETING CARD  (21520, purchased 24)
- GRAND CHOCOLATECANDLE (72741, purchased 18)
- FAIRY CAKES NOTEBOOK A6 SIZE (84535B, purchased 16)
- ASSORTED CAKES FRIDGE MAGNETS (85216, purchased 12)
- FAIRY CAKE BIRTHDAY CANDLE SET (37495, purchased 12)
- CARD BILLBOARD FONT (22983, purchased 12)
- CARD MOTORBIKE SANTA (22816, purchased 12)
- PARTY CONE CHRISTMAS DECORATION  (22130, purchased 12)
- CHOC TRUFFLE GOLD TRINKET POT  (22067, purchased 12)
- CHRISTMAS PUDDING TRINKET POT  (22065, purchased 12)
- PINK VINTAGE PAISLEY PICNIC BAG (21933, purchased 10)
- SCANDINAVIAN PAISLEY PICNIC BAG (21932, purchased 10)
- TEA TIME KITCHEN APRON (47567B, purchased 9)
- MINI CAKE STAND WITH HANGING CAKES (37446, purchased 8)
- MINI CAKE STAND  HANGING STRAWBERY (22055, purchased 8)
- SMALL GLASS HEART TRINKET POT (21314, purchased 8)
- POSTAGE (POST, purchased 7)
... and 13 more products

CANDIDATE LIST (suggested — strongly prefer these, but you may pick other catalog products if you have a clear reason):
- WORLD WAR 2 GLIDERS ASSTD DESIGNS (84077)
- JUMBO BAG RED RETROSPOT (85099B)
- WHITE HANGING HEART T-LIGHT HOLDER (85123A)
- PACK OF 72 RETROSPOT CAKE CASES (21212)
- PAPER CRAFT , LITTLE BIRDIE (23843)
- ASSORTED COLOUR BIRD ORNAMENT (84879)
- SMALL POPCORN HOLDER (22197)
- MEDIUM CERAMIC TOP STORAGE JAR (23166)
- BROCADE RING PURSE  (17003)
- PACK OF 60 PINK PAISLEY CAKE CASES (21977)
... (40 more candidates)
```

---

## `llm_cf` — adds `Customers with similar purchase patterns also bought` (~3,870 chars; +~470 vs `llm_base`)

Same `Customer purchase history` block as `llm_base`, then a CF-neighbors block before the candidate list:

```
Customers with similar purchase patterns also bought:
- COLOUR GLASS. STAR T-LIGHT HOLDER (71477)
- TEA TIME CAKE STAND IN GIFT BOX (37503)
- 36 DOILIES VINTAGE CHRISTMAS (22950)
- SET OF 4 FAIRY CAKES COASTERS  (84510B)
- WOOLLY HAT SOCK GLOVE ADVENT STRING (35832)
- CERAMIC BOWL WITH LOVE HEART DESIGN (22062)
- PENS ASSORTED FUNKY JEWELED  (22608)
- PINK DOUGHNUT TRINKET POT  (22064)
- CAKE STAND LOVEBIRD 2 TIER WHITE (22220)
- CANDY SPOT EGG WARMER HARE (85093)
```

CF neighbors are also merged into the candidate pool (deduplicated, capped at 50).

---

## Reproducing this

```bash
.venv/bin/python -c "
import sqlite3, pandas as pd
from src.llm import build_prompt
con = sqlite3.connect('data/retail.db'); cid = '12421'
all_codes = set(pd.read_sql('SELECT stock_code FROM products', con)['stock_code'].astype(str).str.upper())
for tier in ['llm_base','llm_cf']:
    p, _ = build_prompt(cid, tier, con, all_codes)
    print(f'\\n===== {tier} ({len(p)} chars) =====\\n{p}')
"
```
