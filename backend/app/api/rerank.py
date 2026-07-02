import time

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.schemas import RerankRequest, RerankResponse, RerankResult
from backend.app.db.session import get_db
from backend.app.db.models import RerankLog
from model.reranker.cross_encoder import cross_encoder_reranker

router = APIRouter()


@router.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest, db: Session = Depends(get_db)):
    start_time = time.perf_counter()

    candidates = [item.model_dump() for item in request.candidates]
    reranked = cross_encoder_reranker.rerank(query=request.query, items=candidates)

    latency_ms = (time.perf_counter() - start_time) * 1000

    scores = [item["rerank_score"] for item in reranked]
    scores_arr = np.array(scores)

    log_entry = RerankLog(
        user_id=request.user_id,
        query=request.query,
        num_candidates=len(candidates),
        top_item_id=reranked[0]["item_id"] if reranked else None,
        top_score=float(scores_arr.max()) if len(scores_arr) > 0 else 0.0,
        min_score=float(scores_arr.min()) if len(scores_arr) > 0 else 0.0,
        max_score=float(scores_arr.max()) if len(scores_arr) > 0 else 0.0,
        mean_score=float(scores_arr.mean()) if len(scores_arr) > 0 else 0.0,
        latency_ms=round(latency_ms, 3),
    )
    db.add(log_entry)
    db.commit()

    results = [
        RerankResult(
            item_id=item["item_id"],
            description=item["description"],
            rerank_score=item["rerank_score"],
        )
        for item in reranked
    ]

    return RerankResponse(
        user_id=request.user_id,
        query=request.query,
        results=results,
        latency_ms=round(latency_ms, 3),
    )
