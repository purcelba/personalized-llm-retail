"""
Phase 10 — Recommendations API
FastAPI server serving pre-computed LLM recommendations from the llm_results table.
Start with: uvicorn src.api:app --reload
"""
from fastapi import FastAPI

app = FastAPI(title="Retail Recommendations API")


@app.get("/health")
def health():
    pass


@app.get("/customers")
def list_customers():
    pass


@app.get("/recommendations/{customer_id}")
def get_recommendations(customer_id: str, tier: str = None):
    pass
