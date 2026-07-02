from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.db.session import engine, Base
from backend.app.db import models
from backend.app.api import retrieval, rerank, feedback

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Two-Tower RecSys API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(retrieval.router, prefix="/api", tags=["retrieval"])
app.include_router(rerank.router, prefix="/api", tags=["rerank"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])


@app.get("/")
def read_root():
    return {"status": "ok", "service": "two-tower-recsys-backend", "phase": "5"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
