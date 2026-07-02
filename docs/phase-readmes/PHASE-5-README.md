# Phase 5 — Real-Time Feedback Stream

## What was built
- JDK 17 and Apache Kafka 3.7.x installed natively on Windows with no Docker
- ZooKeeper and Kafka broker running as separate processes with local log directories
- feedback-events Kafka topic with 1 partition
- KafkaProducerService singleton wrapping confluent-kafka Producer, used by the feedback endpoint
- POST /api/feedback endpoint that publishes interaction events (click, skip, dwell time) to Kafka and returns immediately
- Standalone kafka_consumer.py process that subscribes to feedback-events, reads messages, and writes each event to the feedback_events SQLite table
- FeedbackEvent SQLAlchemy model added to the DB layer

## How to run
Start in this order, one per terminal:
Terminal 1: C:\kafka\bin\windows\zookeeper-server-start.bat C:\kafka\config\zookeeper.properties
Terminal 2: C:\kafka\bin\windows\kafka-server-start.bat C:\kafka\config\server.properties
Terminal 3: uvicorn backend.app.main:app --reload --port 8000
Terminal 4: python streaming\kafka_consumer.py

## Key technical decisions
- Feedback endpoint publishes to Kafka and returns immediately rather than writing to DB directly, keeping API latency minimal and decoupling ingestion from persistence
- Consumer uses earliest offset reset so no events are missed if the consumer restarts
- Consumer group ID set explicitly so Kafka tracks offsets and the consumer resumes from where it left off on restart
- Kafka and ZooKeeper log directories moved from /tmp to C:/kafka/* to avoid data loss on Windows temp cleanup

## Files created
- backend/app/services/kafka_producer.py
- backend/app/api/feedback.py
- backend/app/db/models.py (updated)
- backend/app/schemas.py (updated)
- backend/app/main.py (updated)
- streaming/kafka_consumer.py
