# Phase 3 — ANN Retrieval Service

## What was built
- SQLite database setup via SQLAlchemy (engine, session factory, base model)
- RetrievalLog table tracking every retrieval request: user_id, top_k, latency_ms, retrieved item IDs, timestamp
- FAISS flat inner-product index built from the item embeddings exported in Phase 2, persisted to disk so it does not need to be rebuilt on every server restart
- UserEmbedderService that loads the trained UserTower weights and computes a live embedding for any user_id on request
- POST /api/retrieve endpoint that computes the user embedding, queries FAISS for top-K nearest items, logs latency to SQLite, and returns ranked item IDs with similarity scores

## How to run
.\venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --port 8000
Then call POST http://localhost:8000/api/retrieve with a JSON body like {"user_id": 1, "top_k": 10}

## Key technical decisions
- IndexFlatIP (exact inner-product search) chosen over an approximate index since the catalog is small (1500 items); this can be swapped for IndexIVFFlat or HNSW later without changing the API surface
- User embeddings computed live per-request rather than precomputed, since the user tower is cheap to run and this keeps the retrieval service stateless with respect to user features
- Retrieval latency logged to SQLite on every request so Phase 7's dashboard has real data to chart from day one
- FAISS index file and SQLite DB excluded from Git since both are regeneratable/local artifacts

## Files created
- backend/app/db/session.py
- backend/app/db/models.py
- backend/app/services/faiss_index.py
- backend/app/services/user_embedder.py
- backend/app/schemas.py
- backend/app/api/retrieval.py
- backend/app/main.py (updated)
- model/artifacts/item_index.faiss (generated, not committed)
- recsys.db (generated, not committed)
