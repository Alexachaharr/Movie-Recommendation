"""Recommendation cache: in-process TTL+LRU by default, Redis when REDIS_URL is set.

Invalidation uses per-user *generation counters*: the cache key contains the model version
and the user's generation, so a new rating (bump generation) or a retrained model (new
version) makes stale entries unreachable instantly - no scans or key deletion required,
and it works the same across multiple workers when Redis is used.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any


class MemoryCache:
    backend = "memory"

    def __init__(self, max_entries: int = 50_000, ttl: int = 600):
        self.max_entries, self.ttl = max_entries, ttl
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._gens: dict[int, int] = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str):
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] < time.monotonic():
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: str, value, ttl: int | None = None) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + (ttl or self.ttl), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def user_gen(self, user_id: int) -> int:
        return self._gens.get(user_id, 0)

    def bump_user(self, user_id: int) -> None:
        with self._lock:
            self._gens[user_id] = self._gens.get(user_id, 0) + 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"backend": self.backend, "entries": len(self._data), "hits": self.hits,
                "misses": self.misses, "hit_rate": round(self.hits / total, 4) if total else 0.0}


class RedisCache:
    backend = "redis"

    def __init__(self, url: str, ttl: int = 600, prefix: str = "mrec:"):
        import redis  # optional dependency
        self.r = redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=1)
        self.r.ping()
        self.ttl, self.prefix = ttl, prefix
        self.hits = self.misses = 0

    def get(self, key: str):
        try:
            raw = self.r.get(self.prefix + key)
        except Exception:
            self.misses += 1
            return None  # cache must never take the service down
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value, ttl: int | None = None) -> None:
        try:
            self.r.setex(self.prefix + key, ttl or self.ttl, json.dumps(value))
        except Exception:
            pass

    def user_gen(self, user_id: int) -> int:
        try:
            v = self.r.get(f"{self.prefix}gen:{user_id}")
            return int(v) if v else 0
        except Exception:
            return int(time.time())  # Redis down -> effectively bypass cache

    def bump_user(self, user_id: int) -> None:
        try:
            self.r.incr(f"{self.prefix}gen:{user_id}")
        except Exception:
            pass

    def clear(self) -> None:
        try:
            for k in self.r.scan_iter(self.prefix + "rec:*"):
                self.r.delete(k)
        except Exception:
            pass

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"backend": self.backend, "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}


def make_cache(redis_url: str | None, max_entries: int, ttl: int):
    if redis_url:
        try:
            return RedisCache(redis_url, ttl=ttl)
        except Exception as exc:  # fall back rather than fail to boot
            print(f"[cache] Redis unavailable ({exc}); using in-memory cache")
    return MemoryCache(max_entries=max_entries, ttl=ttl)
