"""End-to-end inference latency benchmark.

Replays the same request trace (users drawn from a Zipf-like distribution, as in real traffic
where a minority of users are very active) through three code paths:

  1. untuned   - ``SELECT *`` for the user's ratings, then one query per recommended movie
                 and per explanation movie (classic N+1), no result cache
  2. tuned SQL - single narrow query served from the covering index, in-memory catalog
  3. tuned SQL + cache - as (2) plus the TTL/LRU (or Redis) recommendation cache

The model computation is identical in all three, so differences come from data access
and caching alone.
"""
from __future__ import annotations

import time

import numpy as np
from sqlalchemy import text

from . import db
from .cache import MemoryCache
from .service import RecommenderService


def _untuned(svc: RecommenderService, user_id: int, n: int) -> dict:
    with svc.engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM ratings WHERE user_id = :u"), {"u": user_id}).fetchall()
        ids = np.array([r.movie_id for r in rows], dtype=np.int64)
        vals = np.array([r.rating for r in rows], dtype=np.float32)
        rec = svc.model.recommend(ids, vals, n=n)
        for item in rec["items"]:
            m = conn.execute(text("SELECT * FROM movies WHERE id = :i"), {"i": item["movie_id"]}).first()
            item["title"], item["year"], item["genres"] = m.title, m.year, m.genres.split("|")
            if item["because_you_liked"] is not None:
                b = conn.execute(text("SELECT * FROM movies WHERE id = :i"), {"i": item["because_you_liked"]}).first()
                item["because_you_liked"] = {"id": b.id, "title": b.title}
    return rec


def _summ(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"mean_ms": round(float(a.mean()), 3), "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3)}


def run(engine, requests: int = 3000, n: int = 10, seed: int = 7) -> dict:
    svc = RecommenderService(engine, cache=MemoryCache(max_entries=100_000, ttl=3600))
    svc.load()
    if svc.model is None:
        raise RuntimeError("train a model first: python -m app.cli train")
    with engine.connect() as conn:
        user_ids = np.array([r[0] for r in conn.execute(text("SELECT id FROM users"))])
        plan = db.explain_user_ratings(conn, int(user_ids[0]))
    rng = np.random.default_rng(seed)
    rank_w = 1.0 / np.arange(1, len(user_ids) + 1) ** 0.9
    trace = rng.choice(rng.permutation(user_ids), size=requests, p=rank_w / rank_w.sum())

    # warm-up (connection pool, OS page cache, numpy)
    for uid in trace[:50]:
        _untuned(svc, int(uid), n)
        svc.recommend(int(uid), n, use_cache=False)

    paths = {
        "untuned": lambda u: _untuned(svc, u, n),
        "tuned_sql": lambda u: svc.recommend(u, n, use_cache=False),
        "tuned_sql_plus_cache": lambda u: svc.recommend(u, n, use_cache=True),
    }
    results = {}
    for name, fn in paths.items():
        ms = []
        for uid in trace:
            t0 = time.perf_counter()
            fn(int(uid))
            ms.append((time.perf_counter() - t0) * 1000)
        results[name] = _summ(ms)

    base = results["untuned"]["mean_ms"]
    for name in ("tuned_sql", "tuned_sql_plus_cache"):
        results[name]["mean_reduction_vs_untuned_pct"] = round(100 * (1 - results[name]["mean_ms"] / base), 1)
        results[name]["p95_reduction_vs_untuned_pct"] = round(
            100 * (1 - results[name]["p95_ms"] / results["untuned"]["p95_ms"]), 1)
    return {
        "requests": requests, "distinct_users": int(len(set(trace.tolist()))), "top_n": n,
        "database": engine.dialect.name, "user_ratings_query_plan": plan,
        "cache_hit_rate": svc.cache.stats()["hit_rate"],
        "results": results,
    }
