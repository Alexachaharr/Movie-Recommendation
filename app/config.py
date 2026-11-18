"""Runtime configuration, read from environment variables (12-factor style)."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _normalize_db_url(url: str) -> str:
    # Heroku/Render/Railway hand out postgres:// or postgresql:// URLs; use the psycopg3 driver.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATABASE_URL = _normalize_db_url(os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'movies.db'}"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
MODEL_PATH = MODEL_DIR / "hybrid_model.npz"

# Caching
REDIS_URL = os.getenv("REDIS_URL") or None
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "50000"))

# Model hyper-parameters
N_FACTORS = int(os.getenv("N_FACTORS", "32"))        # explicit-rating SVD rank
N_IMPLICIT = int(os.getenv("N_IMPLICIT", "16"))      # implicit (co-watch) SVD rank
REG_ITEM = float(os.getenv("REG_ITEM", "25"))
REG_USER = float(os.getenv("REG_USER", "10"))
CF_SHRINK = float(os.getenv("CF_SHRINK", "15"))  # ratings needed before CF dominates the blend

# Bootstrap (used when the database is empty on first start)
SEED_USERS = int(os.getenv("SEED_USERS", "10000"))
SEED_MOVIES = int(os.getenv("SEED_MOVIES", "2500"))
SEED_MEAN_RATINGS = int(os.getenv("SEED_MEAN_RATINGS", "60"))

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN") or None
MODEL_RELOAD_CHECK_SECONDS = float(os.getenv("MODEL_RELOAD_CHECK_SECONDS", "30"))
