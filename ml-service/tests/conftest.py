"""Shared test setup.

Two protections, both applied before any `app` module is imported:

* Tests never touch the developer's real data. The integration tests TRUNCATE
  tables, so they run against a separate `<db>_test` database (created on
  demand) rather than the one the app uses. They refuse outright to run
  against a non-local server.
* ml-service/.env may configure a real LLM provider for local use. Tests must
  never depend on (or spend) a live API key, so every test runs against the
  deterministic keyless extractive provider unless it opts in explicitly.
"""

import os

import psycopg
import pytest
from sqlalchemy.engine import make_url

from app.config import get_settings

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", None, ""}


def _point_at_test_database() -> None:
    url = make_url(get_settings().database_url)
    if url.host not in _LOCAL_HOSTS:
        pytest.exit(
            f"Refusing to run tests against non-local database host {url.host!r}: "
            "the integration tests TRUNCATE tables.",
            returncode=2,
        )
    if not url.database or url.database.endswith("_test"):
        return

    test_name = f"{url.database}_test"
    try:
        with psycopg.connect(
            host=url.host,
            port=url.port,
            user=url.username,
            password=url.password,
            dbname="postgres",
            autocommit=True,
        ) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (test_name,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{test_name}"')
    except psycopg.OperationalError:
        return  # no Postgres: integration tests skip themselves, unit tests don't need one

    test_url = url.set(database=test_name)
    os.environ["DATABASE_URL"] = test_url.render_as_string(hide_password=False)
    get_settings.cache_clear()


_point_at_test_database()


@pytest.fixture(autouse=True)
def _force_extractive_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
