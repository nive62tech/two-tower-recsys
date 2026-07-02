from pydantic import BaseModel
from typing import List, Optional


class RetrieveRequest(BaseModel):
    user_id: int
    top_k: int = 20


class RetrieveResponse(BaseModel):
    user_id: int
    item_ids: List[int]
    scores: List[float]
    latency_ms: float


class RerankItem(BaseModel):
    item_id: int
    description: str


class RerankRequest(BaseModel):
    user_id: int
    query: str
    candidates: List[RerankItem]


class RerankResult(BaseModel):
    item_id: int
    description: str
    rerank_score: float


class RerankResponse(BaseModel):
    user_id: int
    query: str
    results: List[RerankResult]
    latency_ms: float


class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    event_type: str
    dwell_seconds: Optional[float] = None
    label: int


class FeedbackResponse(BaseModel):
    status: str
    user_id: int
    item_id: int
    event_type: str
