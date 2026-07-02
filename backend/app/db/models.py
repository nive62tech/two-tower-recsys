from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime

from backend.app.db.session import Base


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    top_k = Column(Integer)
    latency_ms = Column(Float)
    retrieved_item_ids = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class RerankLog(Base):
    __tablename__ = "rerank_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    query = Column(String)
    num_candidates = Column(Integer)
    top_item_id = Column(Integer)
    top_score = Column(Float)
    min_score = Column(Float)
    max_score = Column(Float)
    mean_score = Column(Float)
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class FeedbackEvent(Base):
    __tablename__ = "feedback_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    item_id = Column(Integer, index=True)
    event_type = Column(String)
    dwell_seconds = Column(Float, nullable=True)
    label = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
