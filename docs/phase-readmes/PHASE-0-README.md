# Phase 0 — Repo Setup & Environment

## What was built
- GitHub repository created and cloned locally
- Full project folder structure scaffolded (data, model, backend, streaming, frontend, scripts, docs)
- Python virtual environment created with all core dependencies installed (FastAPI, PyTorch, FAISS, sentence-transformers, confluent-kafka)
- Hello-world FastAPI backend running with CORS enabled for the frontend
- Hello-world Next.js + TypeScript + Tailwind frontend running

## How to run
Backend:
.\venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --port 8000

Frontend:
cd frontend
npm run dev

## Key technical decisions
- No Docker used anywhere; all tools installed natively on Windows
- SQLite chosen over a hosted DB for zero-setup persistence
- FAISS CPU build chosen since no paid GPU is available locally
- CORS restricted to localhost:3000 to match the Next.js dev server

## Files created
- .gitignore
- requirements.txt
- backend/app/__init__.py
- backend/app/api/__init__.py
- backend/app/services/__init__.py
- backend/app/db/__init__.py
- backend/app/main.py
- frontend/ (Next.js app scaffold)
- frontend/app/page.tsx
