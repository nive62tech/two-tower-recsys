from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.models import RetrievalLog, RerankLog, EmbeddingSnapshot, FeedbackEvent

router = APIRouter()

ARTIFACTS_DIR = Path("model/artifacts")
SNAPSHOTS_DIR = ARTIFACTS_DIR / "snapshots"


@router.get("/metrics/retrieval-latency")
def get_retrieval_latency(limit: int = 50, db: Session = Depends(get_db)):
    rows = (
        db.query(RetrievalLog.created_at, RetrievalLog.latency_ms)
        .order_by(RetrievalLog.created_at.asc())
        .limit(limit)
        .all()
    )
    return [{"timestamp": str(r.created_at), "latency_ms": round(r.latency_ms, 3)} for r in rows]


@router.get("/metrics/rerank-scores")
def get_rerank_scores(limit: int = 50, db: Session = Depends(get_db)):
    rows = (
        db.query(RerankLog.created_at, RerankLog.min_score, RerankLog.max_score, RerankLog.mean_score)
        .order_by(RerankLog.created_at.asc())
        .limit(limit)
        .all()
    )
    return [
        {
            "timestamp": str(r.created_at),
            "min_score": round(r.min_score or 0, 4),
            "max_score": round(r.max_score or 0, 4),
            "mean_score": round(r.mean_score or 0, 4),
        }
        for r in rows
    ]


@router.get("/metrics/embedding-snapshots")
def get_embedding_snapshots(db: Session = Depends(get_db)):
    rows = (
        db.query(EmbeddingSnapshot)
        .order_by(EmbeddingSnapshot.created_at.asc())
        .all()
    )
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp,
            "snapshot_file": r.snapshot_file,
            "num_items": r.num_items,
            "embedding_dim": r.embedding_dim,
        }
        for r in rows
    ]


@router.get("/metrics/embedding-pca")
def get_embedding_pca(snapshot_file: str = None):
    import pandas as pd
    from sklearn.decomposition import PCA

    if snapshot_file:
        emb_path = SNAPSHOTS_DIR / snapshot_file
    else:
        emb_path = ARTIFACTS_DIR / "item_embeddings.npy"

    if not emb_path.exists():
        return {"error": "Embedding file not found", "points": []}

    embeddings = np.load(str(emb_path)).astype("float32")

    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    items_df = pd.read_csv("data/raw/items.csv")
    categories = items_df["category"].tolist()

    points = []
    for i, (x, y) in enumerate(coords):
        points.append({
            "item_id": items_df["item_id"].iloc[i],
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "category": categories[i] if i < len(categories) else "unknown",
        })

    return {
        "snapshot_file": str(emb_path.name),
        "num_items": len(points),
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "points": points,
    }


@router.get("/metrics/summary")
def get_summary(db: Session = Depends(get_db)):
    total_retrievals = db.query(func.count(RetrievalLog.id)).scalar()
    avg_retrieval_ms = db.query(func.avg(RetrievalLog.latency_ms)).scalar()
    total_reranks = db.query(func.count(RerankLog.id)).scalar()
    avg_rerank_ms = db.query(func.avg(RerankLog.latency_ms)).scalar()
    total_feedback = db.query(func.count(FeedbackEvent.id)).scalar()
    total_snapshots = db.query(func.count(EmbeddingSnapshot.id)).scalar()

    return {
        "total_retrievals": total_retrievals or 0,
        "avg_retrieval_latency_ms": round(avg_retrieval_ms or 0, 3),
        "total_reranks": total_reranks or 0,
        "avg_rerank_latency_ms": round(avg_rerank_ms or 0, 3),
        "total_feedback_events": total_feedback or 0,
        "total_embedding_snapshots": total_snapshots or 0,
    }
