# Phase 4 — Cross-Encoder Reranker

## What was built
- CrossEncoderReranker service loading cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers
- Scores every (query, item description) pair and reorders candidates by score
- RerankLog table in SQLite tracking per-request score statistics: top score, min, max, mean, latency
- POST /api/rerank endpoint wiring the cross-encoder into the API with full SQLite logging
- Full two-stage pipeline verified end to end: FAISS retrieval feeds candidates directly into the cross-encoder reranker

## How to run
.\venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --port 8000
Call POST http://localhost:8000/api/retrieve to get candidates, then feed them into POST http://localhost:8000/api/rerank with a query string.

## Key technical decisions
- cross-encoder/ms-marco-MiniLM-L-6-v2 chosen for being the smallest cross-encoder in the sentence-transformers library that still performs well on relevance scoring, CPU-friendly with no GPU required
- Score distribution statistics (min, max, mean) logged per request rather than individual item scores, to keep the DB lightweight while still giving the dashboard enough signal to plot distributions
- Reranker operates on the candidate shortlist only (top-K from FAISS, typically 10-20 items), never the full catalog, keeping latency acceptable on CPU

## Files created
- model/reranker/__init__.py
- model/reranker/cross_encoder.py
- backend/app/api/rerank.py
- backend/app/db/models.py (updated)
- backend/app/schemas.py (updated)
- backend/app/main.py (updated)
