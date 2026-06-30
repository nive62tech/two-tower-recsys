from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime

from backend.app.db.session import Base


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    top_k = Column(Integer)
    latency_ms = Column(Float)
    retrieved_item_ids = Column(String)  # comma-separated
    created_at = Column(DateTime, default=datetime.utcnow)
