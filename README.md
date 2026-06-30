# Two-Tower Retrieval + Real-Time Reranking System with Feedback Loops

A production-style recommendation system built the way YouTube, TikTok, and
Spotify actually do it: a two-tower neural network learns user and item
embeddings, a FAISS ANN index retrieves top-K candidates in milliseconds, a
cross-encoder reranker scores and reorders them, and real-time user
interactions (clicks, dwell time, skips) flow back through a Kafka stream to
continuously update embeddings via online learning. A live dashboard surfaces
retrieval latency, reranking score distributions, and embedding space
evolution over time.

## Why This Project Matters

The two-tower retrieval + reranking pattern is the backbone of large-scale
recommendation systems at Google, Meta, Pinterest, and Netflix. Most ML
portfolio projects stop at "train a model, report accuracy." This project
goes further by implementing the full production loop:

- **Two-stage retrieval**: ANN candidate generation followed by precise
  cross-encoder reranking â€” the same pattern used to serve recommendations
  over catalogs with millions of items in milliseconds.
- **Real-time feedback ingestion**: a Kafka-based event stream captures live
  user interactions instead of relying on static offline datasets.
- **Online learning**: embeddings update incrementally from the live stream,
  not just from periodic offline retraining.
- **Observability**: a dashboard exposes system internals (latency, score
  distributions, embedding drift) that are normally invisible â€” exactly what
  separates a research notebook from a production system.

## Tech Stack

**Model Training:** PyTorch, sentence-transformers (cross-encoder), trained
on Google Colab GPU, exported via TorchScript/ONNX

**Retrieval & Serving:** FAISS (CPU), FastAPI, Uvicorn, SQLite, SQLAlchemy

**Real-Time Streaming:** Apache Kafka (native Windows install), confluent-kafka-python

**Online Learning:** Incremental PyTorch SGD updates triggered from the Kafka consumer

**Frontend:** Next.js, TypeScript, Tailwind CSS, Recharts, Plotly.js/D3

**Environment:** Windows, VSCode + PowerShell, Git/GitHub, no Docker, no paid GPU/cloud

## Folder Structure

\`\`\`
two-tower-recsys/
â”œâ”€â”€ data/                # synthetic data generation + processed splits
â”œâ”€â”€ model/                # two-tower model, reranker, online learning, Colab notebook
â”œâ”€â”€ backend/              # FastAPI app: retrieval, rerank, feedback, metrics APIs
â”œâ”€â”€ streaming/            # Kafka consumer + online training worker
â”œâ”€â”€ frontend/              # Next.js dashboard
â”œâ”€â”€ scripts/               # local orchestration helpers
â””â”€â”€ docs/                  # phase READMEs, architecture diagram
\`\`\`

## Phase Progress

| Phase | Name | Covers | Status |
|-------|------|--------|--------|
| 0 | Repo Setup & Environment | Git init, folder structure, Python/Node env setup | Complete |
| 1 | Synthetic Data Pipeline | Synthetic users/items/interactions, train/val/test splits | Complete |
| 2 | Two-Tower Model Training | UserTower/ItemTower architecture, training on Colab, export | Complete |
| 3 | ANN Retrieval Service | FAISS index build, FastAPI retrieve endpoint, latency logging | Pending |
| 4 | Cross-Encoder Reranker | sentence-transformers reranker, FastAPI rerank endpoint | Pending |
| 5 | Real-Time Feedback Stream | Kafka native install, feedback endpoint, consumer service | Pending |
| 6 | Online Learning Loop | Incremental SGD updates, embedding snapshot versioning | Pending |
| 7 | Dashboard | Next.js dashboard: latency, score distribution, embedding evolution | Pending |
| 8 | Integration & End-to-End Test | Full pipeline test, load testing, final docs and polish | Pending |
