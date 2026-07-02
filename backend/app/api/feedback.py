from fastapi import APIRouter
from backend.app.schemas import FeedbackRequest, FeedbackResponse
from backend.app.services.kafka_producer import kafka_producer_service

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    event = {
        "user_id": request.user_id,
        "item_id": request.item_id,
        "event_type": request.event_type,
        "dwell_seconds": request.dwell_seconds,
        "label": request.label,
    }
    kafka_producer_service.produce_feedback(event)

    return FeedbackResponse(
        status="published",
        user_id=request.user_id,
        item_id=request.item_id,
        event_type=request.event_type,
    )
