import json
from confluent_kafka import Producer

KAFKA_CONFIG = {
    "bootstrap.servers": "localhost:9092",
}

FEEDBACK_TOPIC = "feedback-events"


class KafkaProducerService:
    def __init__(self):
        self.producer = Producer(KAFKA_CONFIG)

    def produce_feedback(self, event: dict):
        self.producer.produce(
            topic=FEEDBACK_TOPIC,
            key=str(event.get("user_id", "")),
            value=json.dumps(event).encode("utf-8"),
            callback=self._delivery_callback,
        )
        self.producer.poll(0)

    def flush(self):
        self.producer.flush()

    @staticmethod
    def _delivery_callback(err, msg):
        if err:
            print(f"[KafkaProducer] Delivery failed: {err}")
        else:
            print(f"[KafkaProducer] Delivered to {msg.topic()} [{msg.partition()}] offset {msg.offset()}")


kafka_producer_service = KafkaProducerService()
