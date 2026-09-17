"""Fixtures for DB-backed integration tests.

Skips the whole directory when Postgres isn't reachable (e.g. a contributor
running `pytest` without `docker compose up -d`), so the fast, DB-free unit
tests elsewhere in tests/ are unaffected either way.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import engine, init_db
from app.main import app


def _db_reachable() -> bool:
    try:
        with engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_db():
    if not _db_reachable():
        pytest.skip("Postgres not reachable at DATABASE_URL - skipping integration tests")
    init_db()


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE documents, chunks, claims, quiz_questions, publications "
                "RESTART IDENTITY CASCADE"
            )
        )
