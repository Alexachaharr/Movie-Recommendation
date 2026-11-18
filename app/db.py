"""Database schema, engine and tuned SQL queries.

Works with SQLite (default, zero-config) and PostgreSQL (set DATABASE_URL).

Query-tuning decisions (these are what keep inference fast):
  * ``ix_ratings_user_cover (user_id, movie_id, rating)`` is a *covering* index, so the
    hot-path query "all ratings of user X" is answered from the index alone
    (SQLite "USING COVERING INDEX", Postgres "Index Only Scan") with no table lookups.
  * The hot path selects only the two columns it needs instead of ``SELECT *``.
  * Movie metadata is read once into an in-memory catalog, so a recommendation request
    issues exactly one SQL query instead of 1 + N look-ups.
  * SQLite runs in WAL mode with a larger page cache and memory-mapped I/O.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Iterator

import numpy as np
from sqlalchemy import (
    BigInteger, Column, Float, ForeignKey, Index, Integer, MetaData, PrimaryKeyConstraint,
    String, Table, create_engine, event, func, select, text,
)
from sqlalchemy.engine import Connection, Engine

from . import config

metadata = MetaData()

movies = Table(
    "movies", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", String(400), nullable=False),
    Column("year", Integer, nullable=True),
    Column("genres", String(400), nullable=False, default=""),
)

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(120), nullable=False),
    Column("created_at", BigInteger, nullable=False),
)

ratings = Table(
    "ratings", metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("movie_id", Integer, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False),
    Column("rating", Float, nullable=False),
    Column("ts", BigInteger, nullable=False),
    PrimaryKeyConstraint("user_id", "movie_id", name="pk_ratings"),
)

Index("ix_ratings_user_cover", ratings.c.user_id, ratings.c.movie_id, ratings.c.rating)
Index("ix_ratings_movie", ratings.c.movie_id)


def _sqlite_pragmas(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA cache_size=-65536")      # 64 MB page cache
    cur.execute("PRAGMA mmap_size=268435456")    # 256 MB memory-mapped I/O
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.close()


def make_engine(url: str | None = None) -> Engine:
    url = url or config.DATABASE_URL
    if url.startswith("sqlite"):
        if ":///" in url and not url.endswith(":memory:"):
            from pathlib import Path
            Path(url.split(":///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
        event.listen(engine, "connect", _sqlite_pragmas)
    else:
        engine = create_engine(url, pool_size=10, max_overflow=20, pool_pre_ping=True)
    return engine


@lru_cache(maxsize=None)
def get_engine() -> Engine:
    return make_engine()


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def now_ts() -> int:
    return int(time.time())


# ---------------------------------------------------------------- hot-path queries

USER_RATINGS_SQL = text("SELECT movie_id, rating FROM ratings WHERE user_id = :uid")


def fetch_user_ratings(conn: Connection, user_id: int) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(USER_RATINGS_SQL, {"uid": user_id}).fetchall()
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.float32)


def user_exists(conn: Connection, user_id: int) -> bool:
    return conn.execute(text("SELECT 1 FROM users WHERE id = :uid"), {"uid": user_id}).first() is not None


def upsert_rating(conn: Connection, user_id: int, movie_id: int, rating: float) -> None:
    values = {"user_id": user_id, "movie_id": movie_id, "rating": float(rating), "ts": now_ts()}
    name = conn.dialect.name
    if name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:  # generic fallback
        conn.execute(ratings.delete().where(
            (ratings.c.user_id == user_id) & (ratings.c.movie_id == movie_id)))
        conn.execute(ratings.insert().values(**values))
        return
    stmt = insert(ratings).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["user_id", "movie_id"],
        set_={"rating": stmt.excluded.rating, "ts": stmt.excluded.ts},
    )
    conn.execute(stmt)


def delete_rating(conn: Connection, user_id: int, movie_id: int) -> int:
    res = conn.execute(ratings.delete().where(
        (ratings.c.user_id == user_id) & (ratings.c.movie_id == movie_id)))
    return res.rowcount or 0


def create_user(conn: Connection, name: str) -> int:
    res = conn.execute(users.insert().values(name=name, created_at=now_ts()))
    return int(res.inserted_primary_key[0])


def user_history(conn: Connection, user_id: int, limit: int) -> list[tuple[int, float, int]]:
    q = text("SELECT movie_id, rating, ts FROM ratings WHERE user_id = :uid "
             "ORDER BY rating DESC, ts DESC LIMIT :lim")
    return [(int(m), float(r), int(t)) for m, r, t in conn.execute(q, {"uid": user_id, "lim": limit})]


def load_catalog(conn: Connection) -> dict[int, dict]:
    rows = conn.execute(select(movies.c.id, movies.c.title, movies.c.year, movies.c.genres))
    return {int(i): {"id": int(i), "title": t, "year": y, "genres": g or ""} for i, t, y, g in rows}


def counts(conn: Connection) -> dict[str, int]:
    return {
        "users": conn.execute(select(func.count()).select_from(users)).scalar_one(),
        "movies": conn.execute(select(func.count()).select_from(movies)).scalar_one(),
        "ratings": conn.execute(select(func.count()).select_from(ratings)).scalar_one(),
    }


def iter_all_ratings(conn: Connection, chunk: int = 200_000) -> Iterator[np.ndarray]:
    """Stream the ratings table in chunks (keeps memory flat for big datasets)."""
    result = conn.execution_options(stream_results=True).execute(
        text("SELECT user_id, movie_id, rating, ts FROM ratings"))
    while True:
        rows = result.fetchmany(chunk)
        if not rows:
            break
        yield np.asarray(rows, dtype=np.float64)


def load_all_ratings(conn: Connection):
    parts = list(iter_all_ratings(conn))
    if not parts:
        raise RuntimeError("ratings table is empty - run `python -m app.cli seed` first")
    a = np.vstack(parts)
    return a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2].astype(np.float32), a[:, 3].astype(np.int64)


def reset_pg_sequences(conn: Connection) -> None:
    """After bulk-loading explicit ids into Postgres, move the SERIAL sequences past them."""
    if conn.dialect.name != "postgresql":
        return
    for table in ("users", "movies"):
        conn.execute(text(
            f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
            f"COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)"))


def analyze(engine) -> None:
    """Refresh planner statistics after bulk loads (Postgres also needs VACUUM to set the
    visibility map, otherwise it cannot use index-only scans on the covering index)."""
    if engine.dialect.name == "postgresql":
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            for t in ("ratings", "movies", "users"):
                conn.execute(text(f"VACUUM ANALYZE {t}"))
    elif engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            conn.execute(text("ANALYZE"))


def explain_user_ratings(conn: Connection, user_id: int = 1) -> str:
    if conn.dialect.name == "sqlite":
        rows = conn.execute(text("EXPLAIN QUERY PLAN SELECT movie_id, rating FROM ratings WHERE user_id = :uid"),
                            {"uid": user_id}).fetchall()
        return "; ".join(str(r[-1]) for r in rows)
    if conn.dialect.name == "postgresql":
        rows = conn.execute(text("EXPLAIN SELECT movie_id, rating FROM ratings WHERE user_id = :uid"),
                            {"uid": user_id}).fetchall()
        return "; ".join(r[0] for r in rows)
    return "n/a"
