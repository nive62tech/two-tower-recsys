# Phase 6 — Online Learning Loop

## What was built
- Incremental SGD update function that pulls recent positive feedback events from SQLite, runs one training step on the two-tower model, and saves updated weights back to disk
- Online trainer worker process that runs every 30 seconds, triggers the incremental update, exports fresh item embeddings, rebuilds the FAISS index, and saves a versioned embedding snapshot
- Snapshot logger that saves a timestamped numpy file of item embeddings after each update and records metadata in the embedding_snapshots SQLite table
- EmbeddingSnapshot SQLAlchemy model added to the DB layer
- Metrics API stub with four endpoints: retrieval latency history, rerank score distributions, embedding snapshot list, and a summary count endpoint — ready for Phase 7's dashboard to consume

## How to run
Start in this order:
Terminal 1: D:\kafka\kafka_2.13-3.7.0\bin\windows\zookeeper-server-start.bat D:\kafka\kafka_2.13-3.7.0\config\zookeeper.properties
Terminal 2: D:\kafka\kafka_2.13-3.7.0\bin\windows\kafka-server-start.bat D:\kafka\kafka_2.13-3.7.0\config\server.properties
Terminal 3: uvicorn backend.app.main:app --reload --port 8000
Terminal 4: python streaming\online_trainer_worker.py

## Key technical decisions
- SGD with momentum chosen over Adam for incremental updates since it is more stable for small batches and does not accumulate large adaptive learning rates that can distort embeddings trained with Adam originally
- Learning rate set to 0.0001 (10x lower than initial training) to avoid catastrophic forgetting of the original embedding space
- FAISS index rebuilt from scratch after each update rather than updated in place, since IndexFlatIP does not support in-place modification and the rebuild cost is negligible at 1500 items
- Snapshots stored as versioned numpy files so Phase 7 can load any two snapshots and compute PCA to visualize embedding drift over time
- Worker update interval set to 30 seconds for development; in production this would be driven by a minimum batch size threshold rather than a fixed interval

## Files created
- model/online_learning/__init__.py
- model/online_learning/incremental_update.py
- streaming/online_trainer_worker.py
- streaming/snapshot_logger.py
- backend/app/db/models.py (updated)
- backend/app/api/metrics.py
- backend/app/main.py (updated)
- model/artifacts/snapshots/ (directory, generated files not committed)
