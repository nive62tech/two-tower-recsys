import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.schemas import RetrieveRequest, RetrieveResponse
from backend.app.services.faiss_index import faiss_index_service
from backend.app.services.user_embedder import user_embedder_service
from backend.app.db.session import get_db
from backend.app.db.models import RetrievalLog

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest, db: Session = Depends(get_db)):
    start_time = time.perf_counter()

    try:
        user_embedding = user_embedder_service.get_user_embedding(request.user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    item_ids, scores = faiss_index_service.search(user_embedding, top_k=request.top_k)

    latency_ms = (time.perf_counter() - start_time) * 1000

    log_entry = RetrievalLog(
        user_id=request.user_id,
        top_k=request.top_k,
        latency_ms=latency_ms,
        retrieved_item_ids=",".join(str(i) for i in item_ids),
    )
    db.add(log_entry)
    db.commit()

    return RetrieveResponse(
        user_id=request.user_id,
        item_ids=item_ids,
        scores=scores,
        latency_ms=round(latency_ms, 3),
    )
