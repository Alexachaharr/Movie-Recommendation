"""Application service: glues the database, the model and the cache together."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from sqlalchemy.engine import Engine

from . import config, db
from .cache import make_cache
from .model import HybridModel


class NotFound(Exception):
    pass


class RecommenderService:
    def __init__(self, engine: Engine, model_path: Path | None = None, cache=None):
        self.engine = engine
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.cache = cache or make_cache(config.REDIS_URL, config.CACHE_MAX_ENTRIES, config.CACHE_TTL_SECONDS)
        self.model: HybridModel | None = None
        self.catalog: dict[int, dict] = {}
        self._search_index: list[tuple[str, int]] = []
        self._model_mtime = 0.0
        self._last_reload_check = 0.0
        self._train_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        with self.engine.connect() as conn:
            self.catalog = db.load_catalog(conn)
        self._search_index = [(m["title"].lower(), mid) for mid, m in self.catalog.items()]
        if self.model_path.exists():
            self.model = HybridModel.load(self.model_path)
            self._model_mtime = self.model_path.stat().st_mtime

    def maybe_reload(self) -> None:
        """Pick up a model retrained by another worker/process (checked at most every N s)."""
        now = time.monotonic()
        if now - self._last_reload_check < config.MODEL_RELOAD_CHECK_SECONDS:
            return
        self._last_reload_check = now
        try:
            mtime = self.model_path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime > self._model_mtime:
            self.load()

    def train(self, save: bool = True) -> dict:
        with self._train_lock:
            t0 = time.time()
            with self.engine.connect() as conn:
                catalog = db.load_catalog(conn)
                u, m, r, _ = db.load_all_ratings(conn)
            model = HybridModel.train(
                u, m, r, catalog, n_factors=config.N_FACTORS, n_implicit=config.N_IMPLICIT,
                reg_item=config.REG_ITEM, reg_user=config.REG_USER, cf_shrink=config.CF_SHRINK)
            if save:
                model.save(self.model_path)
                self._model_mtime = self.model_path.stat().st_mtime
            self.catalog = catalog
            self._search_index = [(mv["title"].lower(), mid) for mid, mv in catalog.items()]
            self.model = model  # atomic reference swap; in-flight requests keep the old model
            return {**model.meta, "total_seconds": round(time.time() - t0, 2)}

    def _require_model(self) -> HybridModel:
        if self.model is None:
            raise RuntimeError("Model not trained yet. Run `python -m app.cli train`.")
        return self.model

    # ------------------------------------------------------------------ helpers
    def movie(self, movie_id: int) -> dict:
        m = self.catalog.get(int(movie_id))
        if m is None:
            raise NotFound(f"movie {movie_id} not found")
        out = {**m, "genres": [g for g in m["genres"].split("|") if g]}
        if self.model is not None and m["id"] in self.model.index:
            i = self.model.index[m["id"]]
            out["avg_rating_shrunk"] = round(float(self.model.mu + self.model.item_bias[i]), 2)
            out["num_ratings"] = int(self.model.rating_count[i])
        return out

    def _brief(self, movie_id: int | None) -> dict | None:
        if movie_id is None or movie_id not in self.catalog:
            return None
        m = self.catalog[movie_id]
        return {"id": m["id"], "title": m["title"]}

    # ------------------------------------------------------------------ use cases
    def recommend(self, user_id: int, n: int = 10, use_cache: bool = True) -> dict:
        self.maybe_reload()
        model = self._require_model()
        key = f"rec:{model.version}:{user_id}:{self.cache.user_gen(user_id)}:{n}"
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return {**hit, "cached": True}

        with self.engine.connect() as conn:
            rated_ids, rated_vals = db.fetch_user_ratings(conn, user_id)
            if len(rated_ids) == 0 and not db.user_exists(conn, user_id):
                raise NotFound(f"user {user_id} not found")

        rec = model.recommend(rated_ids, rated_vals, n=n)
        for item in rec["items"]:
            m = self.catalog.get(item["movie_id"], {})
            item["title"] = m.get("title")
            item["year"] = m.get("year")
            item["genres"] = [g for g in m.get("genres", "").split("|") if g]
            item["because_you_liked"] = self._brief(item["because_you_liked"])
        strategy = ("popularity (cold start)" if rec["n_ratings_used"] == 0
                    else "hybrid (content-leaning)" if rec["cf_weight"] < 0.5 else "hybrid (CF-leaning)")
        result = {"user_id": user_id, "model_version": model.version, "strategy": strategy, **rec}
        if use_cache:
            self.cache.set(key, result)
        return {**result, "cached": False}

    def similar(self, movie_id: int, n: int = 10) -> dict:
        self.maybe_reload()
        model = self._require_model()
        if int(movie_id) not in self.catalog:
            raise NotFound(f"movie {movie_id} not found")
        key = f"sim:{model.version}:{movie_id}:{n}"
        hit = self.cache.get(key)
        if hit is None:
            hit = [{**s, **self._brief(s["movie_id"]), "genres": self.catalog[s["movie_id"]]["genres"].split("|")}
                   for s in model.similar(movie_id, n)]
            self.cache.set(key, hit)
        return {"movie": self._brief(int(movie_id)), "similar": hit}

    def rate(self, user_id: int, movie_id: int, rating: float) -> None:
        if int(movie_id) not in self.catalog:
            raise NotFound(f"movie {movie_id} not found")
        with self.engine.begin() as conn:
            if not db.user_exists(conn, user_id):
                raise NotFound(f"user {user_id} not found")
            db.upsert_rating(conn, user_id, movie_id, rating)
        self.cache.bump_user(user_id)

    def unrate(self, user_id: int, movie_id: int) -> bool:
        with self.engine.begin() as conn:
            deleted = db.delete_rating(conn, user_id, movie_id)
        self.cache.bump_user(user_id)
        return bool(deleted)

    def create_user(self, name: str) -> dict:
        with self.engine.begin() as conn:
            uid = db.create_user(conn, name)
        return {"id": uid, "name": name}

    def history(self, user_id: int, limit: int = 50) -> dict:
        with self.engine.connect() as conn:
            if not db.user_exists(conn, user_id):
                raise NotFound(f"user {user_id} not found")
            rows = db.user_history(conn, user_id, limit)
            total = len(db.fetch_user_ratings(conn, user_id)[0])
        items = [{**(self._brief(m) or {"id": m, "title": None}), "rating": r, "ts": t} for m, r, t in rows]
        return {"user_id": user_id, "total_ratings": total, "ratings": items}

    def search(self, q: str, limit: int = 20) -> list[dict]:
        q = q.strip().lower()
        if not q:
            return []
        hits = [mid for title, mid in self._search_index if q in title]
        if self.model is not None:
            idx = self.model.index
            hits.sort(key=lambda mid: (not self.catalog[mid]["title"].lower().startswith(q),
                                       -float(self.model.popularity[idx[mid]]) if mid in idx else 0.0))
        return [self.movie(mid) for mid in hits[:limit]]

    def stats(self) -> dict:
        with self.engine.connect() as conn:
            c = db.counts(conn)
        return {"database": {**c, "dialect": self.engine.dialect.name},
                "model": self.model.meta if self.model else None,
                "cache": self.cache.stats()}
