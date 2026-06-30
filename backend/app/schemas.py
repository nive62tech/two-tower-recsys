from pydantic import BaseModel
from typing import List


class RetrieveRequest(BaseModel):
    user_id: int
    top_k: int = 20


class RetrieveResponse(BaseModel):
    user_id: int
    item_ids: List[int]
    scores: List[float]
    latency_ms: float
