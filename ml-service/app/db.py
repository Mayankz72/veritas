from collections.abc import Generator

from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


@event.listens_for(engine, "connect")
def _register_vector_type(dbapi_connection, connection_record) -> None:
    # Every new physical connection needs both the extension (idempotent,
    # cheap) and the psycopg vector type adapter registered on it, since the
    # pool may hand out a connection that predates the extension existing.
    with dbapi_connection.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    dbapi_connection.commit()
    register_vector(dbapi_connection)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    import app.models  # noqa: F401  (register models on Base before create_all)

    Base.metadata.create_all(bind=engine)
