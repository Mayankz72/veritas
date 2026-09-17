# Veritas — Evidence-Grounded Research Paper Studio

Turns research papers into interactive, verifiable workspaces — every generated
claim links back to the exact page and quote it rests on, backed by a real
retrieval + grounding-verification pipeline (not just an LLM asserting citations).

See [ROADMAP.md](./ROADMAP.md) for the full build plan and current status.

## Structure

```
app/            Next.js 16 (TypeScript, App Router) — the web app
ml-service/     FastAPI (Python) — PDF parsing, embeddings, retrieval, grounding eval
packages/shared/  Schema documentation shared between the two services
docker-compose.yml  Postgres + pgvector for local dev
```

## Local development

```bash
# 1. Database
docker compose up -d

# 2. ML service
cd ml-service
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# 3. Web app
cd app
npm install
npm run dev
```

App runs at `http://localhost:3000`, ML service at `http://localhost:8000`.
