from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.db.session import get_db
from backend.app.db.models import RetrievalLog, RerankLog, EmbeddingSnapshot

router = APIRouter()


@router.get("/metrics/retrieval-latency")
def get_retrieval_latency(limit: int = 50, db: Session = Depends(get_db)):
    rows = (
        db.query(RetrievalLog.created_at, RetrievalLog.latency_ms)
        .order_by(RetrievalLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [{"timestamp": str(r.created_at), "latency_ms": r.latency_ms} for r in rows]


@router.get("/metrics/rerank-scores")
def get_rerank_scores(limit: int = 50, db: Session = Depends(get_db)):
    rows = (
        db.query(RerankLog.created_at, RerankLog.min_score, RerankLog.max_score, RerankLog.mean_score)
        .order_by(RerankLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "timestamp": str(r.created_at),
            "min_score": r.min_score,
            "max_score": r.max_score,
            "mean_score": r.mean_score,
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


@router.get("/metrics/summary")
def get_summary(db: Session = Depends(get_db)):
    total_retrievals = db.query(func.count(RetrievalLog.id)).scalar()
    avg_retrieval_ms = db.query(func.avg(RetrievalLog.latency_ms)).scalar()
    total_reranks = db.query(func.count(RerankLog.id)).scalar()
    avg_rerank_ms = db.query(func.avg(RerankLog.latency_ms)).scalar()
    total_feedback = db.query(func.count).scalar() if False else None
    total_snapshots = db.query(func.count(EmbeddingSnapshot.id)).scalar()

    from backend.app.db.models import FeedbackEvent
    total_feedback = db.query(func.count(FeedbackEvent.id)).scalar()

    return {
        "total_retrievals": total_retrievals,
        "avg_retrieval_latency_ms": round(avg_retrieval_ms or 0, 3),
        "total_reranks": total_reranks,
        "avg_rerank_latency_ms": round(avg_rerank_ms or 0, 3),
        "total_feedback_events": total_feedback,
        "total_embedding_snapshots": total_snapshots,
    }
