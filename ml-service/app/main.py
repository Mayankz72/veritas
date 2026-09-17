from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
from app.routers.claims import router as claims_router
from app.routers.documents import router as documents_router
from app.routers.quiz import router as quiz_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Veritas ML Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Matches any localhost/127.0.0.1 port, not just :3000 - `next dev`
    # doesn't always land on 3000 (e.g. it's silently unusable on Windows
    # when Hyper-V/WSL has reserved it via netsh's dynamic port exclusion
    # range, which happened during development of this project).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(claims_router)
app.include_router(quiz_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
