# Skill: LLM Prompts

## Purpose
Assemble context-enriched prompts for each tier and call the LLM API, with checkpointing
so runs are resumable.

## Required Project Config
| Key | Used for |
|---|---|
| `model_id` | Full run model |
| `dev_model_id` | Smoke test model (cheap/fast) |
| `temperature` | API call parameter |
| `max_tokens` | API call parameter |
| `top_k` | Number of recommendations to request |
| `checkpoint_every` | How often to log progress to console |

## Prompt structure (all tiers)

```
System: You are a product recommendation assistant for an online retailer.
        Recommend exactly {K} products the customer has NOT already purchased.
        Respond with a JSON object: {"recommendations": ["PROD1", ...], "rationale": "..."}

User: [tier-specific context block]
```

## Context blocks by group
Build each block by joining data from the database using customer_id.

**`llm_base` — purchase history only**
```
Customer purchase history:
- {product_description} (purchased {n} times)
...
```

**`llm_cf` — + CF neighbor products**
```
[llm_base block]

Customers with similar purchase patterns also bought: {cf_neighbor_products}
```

## API call pattern
```python
client = anthropic.Anthropic()
response = client.messages.create(
    model=model_id,       # from project config
    max_tokens=max_tokens, # from project config
    temperature=temperature, # from project config
    messages=[{"role": "user", "content": prompt}],
    system=system_prompt
)
```

## Checkpointing
- Check db for existing (customer_id, tier, run_id) before calling API
- Save result to db immediately after each call (not in batches)
- Log checkpoint count to console every N calls (N from project config)

## Response parsing
- Parse JSON from response.content[0].text
- On parse failure: log customer_id + raw response, insert null recommendations, continue
- Never abort the run on a single parse failure

## Rate limiting
- Add a small sleep between calls if hitting rate limits (start at 0.5s, increase if needed)
- Do not hardcode sleep — check response headers or catch RateLimitError
