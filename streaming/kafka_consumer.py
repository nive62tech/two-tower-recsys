import json
import sys
from datetime import datetime

from confluent_kafka import Consumer, KafkaError
from sqlalchemy.orm import Session

from backend.app.db.session import engine, Base, SessionLocal
from backend.app.db.models import FeedbackEvent

Base.metadata.create_all(bind=engine)

KAFKA_CONFIG = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "feedback-consumer-group",
    "auto.offset.reset": "earliest",
}

FEEDBACK_TOPIC = "feedback-events"
POLL_TIMEOUT = 1.0


def consume():
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe([FEEDBACK_TOPIC])
    print(f"[Consumer] Subscribed to topic: {FEEDBACK_TOPIC}")
    print("[Consumer] Waiting for messages... Press Ctrl+C to stop.")

    try:
        while True:
            msg = consumer.poll(timeout=POLL_TIMEOUT)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"[Consumer] Reached end of partition {msg.partition()}")
                else:
                    print(f"[Consumer] Error: {msg.error()}")
                continue

            try:
                event = json.loads(msg.value().decode("utf-8"))
                print(f"[Consumer] Received: user={event.get('user_id')} item={event.get('item_id')} type={event.get('event_type')}")

                db: Session = SessionLocal()
                try:
                    feedback_entry = FeedbackEvent(
                        user_id=event.get("user_id"),
                        item_id=event.get("item_id"),
                        event_type=event.get("event_type"),
                        dwell_seconds=event.get("dwell_seconds"),
                        label=event.get("label"),
                        created_at=datetime.utcnow(),
                    )
                    db.add(feedback_entry)
                    db.commit()
                    print(f"[Consumer] Logged to DB: feedback_events id={feedback_entry.id}")
                finally:
                    db.close()

            except Exception as e:
                print(f"[Consumer] Failed to process message: {e}")

    except KeyboardInterrupt:
        print("[Consumer] Shutting down...")
    finally:
        consumer.close()


if __name__ == "__main__":
    consume()
