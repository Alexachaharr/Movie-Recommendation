"""FastAPI application: REST API + web UI."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import config, db
from .service import NotFound, RecommenderService

STATIC_DIR = Path(__file__).parent / "static"
_latency: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = db.get_engine()
    db.init_db(engine)
    svc = RecommenderService(engine)
    svc.load()
    if svc.model is None:
        with engine.connect() as conn:
            has_data = db.counts(conn)["ratings"] > 0
        if has_data:
            print("[startup] no model found - training from database")
            svc.train()
        else:
            print("[startup] database is empty - run `python -m app.cli bootstrap`")
    app.state.svc = svc
    yield
    engine.dispose()


app = FastAPI(
    title="Personalised Movie Recommender",
    version="1.0.0",
    description="Hybrid (collaborative + content-based) movie recommendations with SQL + caching.",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-ms"] = f"{ms:.2f}"
    route = request.scope.get("route")
    if route is not None and request.url.path.startswith("/api"):
        _latency[f"{request.method} {route.path}"].append(ms)
    return response


@app.exception_handler(NotFound)
async def _not_found(_: Request, exc: NotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def _runtime(_: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def svc(request: Request) -> RecommenderService:
    return request.app.state.svc


# ---------------------------------------------------------------- schemas
class RatingIn(BaseModel):
    rating: float = Field(..., ge=0.5, le=5.0, examples=[4.5])

    @field_validator("rating")
    @classmethod
    def half_steps(cls, v: float) -> float:
        if abs(v * 2 - round(v * 2)) > 1e-9:
            raise ValueError("rating must be in steps of 0.5")
        return v


class UserIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


# ---------------------------------------------------------------- routes
@app.get("/health", tags=["ops"])
def health(s: RecommenderService = Depends(svc)):
    s.maybe_reload()
    return {"status": "ok" if s.model else "degraded", "model_loaded": s.model is not None,
            "model_version": s.model.version if s.model else None}


@app.get("/api/stats", tags=["ops"])
def stats(s: RecommenderService = Depends(svc)):
    return s.stats()


@app.get("/api/metrics", tags=["ops"])
def metrics(s: RecommenderService = Depends(svc)):
    out = {}
    for route, vals in _latency.items():
        a = np.fromiter(vals, dtype=float)
        out[route] = {"count": len(a), "p50_ms": round(float(np.percentile(a, 50)), 2),
                      "p95_ms": round(float(np.percentile(a, 95)), 2), "mean_ms": round(float(a.mean()), 2)}
    return {"latency": out, "cache": s.cache.stats()}


@app.get("/api/users/{user_id}/recommendations", tags=["recommendations"])
def recommendations(user_id: int, n: int = Query(10, ge=1, le=100), s: RecommenderService = Depends(svc)):
    return s.recommend(user_id, n)


@app.get("/api/users/{user_id}/ratings", tags=["users"])
def user_ratings(user_id: int, limit: int = Query(50, ge=1, le=1000), s: RecommenderService = Depends(svc)):
    return s.history(user_id, limit)


@app.put("/api/users/{user_id}/ratings/{movie_id}", tags=["users"])
def put_rating(user_id: int, movie_id: int, body: RatingIn, s: RecommenderService = Depends(svc)):
    s.rate(user_id, movie_id, body.rating)
    return {"user_id": user_id, "movie_id": movie_id, "rating": body.rating}


@app.delete("/api/users/{user_id}/ratings/{movie_id}", tags=["users"])
def delete_rating(user_id: int, movie_id: int, s: RecommenderService = Depends(svc)):
    if not s.unrate(user_id, movie_id):
        raise HTTPException(404, "rating not found")
    return {"deleted": True}


@app.post("/api/users", status_code=201, tags=["users"])
def create_user(body: UserIn, s: RecommenderService = Depends(svc)):
    return s.create_user(body.name)


@app.get("/api/movies/search", tags=["movies"])
def search(q: str = Query(..., min_length=1, max_length=100), limit: int = Query(20, ge=1, le=100),
           s: RecommenderService = Depends(svc)):
    return {"query": q, "results": s.search(q, limit)}


@app.get("/api/movies/{movie_id}", tags=["movies"])
def movie(movie_id: int, s: RecommenderService = Depends(svc)):
    return s.movie(movie_id)


@app.get("/api/movies/{movie_id}/similar", tags=["movies"])
def similar(movie_id: int, n: int = Query(10, ge=1, le=50), s: RecommenderService = Depends(svc)):
    return s.similar(movie_id, n)


@app.post("/api/admin/retrain", tags=["ops"])
def retrain(x_admin_token: str | None = Header(None), s: RecommenderService = Depends(svc)):
    if not config.ADMIN_TOKEN:
        raise HTTPException(403, "retraining over HTTP is disabled (set ADMIN_TOKEN)")
    if x_admin_token != config.ADMIN_TOKEN:
        raise HTTPException(401, "invalid admin token")
    return s.train()


# ---------------------------------------------------------------- UI
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
