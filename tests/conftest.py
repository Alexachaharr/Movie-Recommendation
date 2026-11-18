import os
import sys
import tempfile
from pathlib import Path

import pytest

_tmp = tempfile.mkdtemp(prefix="mrec-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["MODEL_DIR"] = f"{_tmp}/models"
os.environ["ADMIN_TOKEN"] = "test-token"
os.environ.pop("REDIS_URL", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, seed  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    engine = db.get_engine()
    db.init_db(engine)
    seed.generate_synthetic(engine, n_users=400, n_movies=300, mean_ratings=40, seed=1, verbose=False)
    from app.main import app
    with TestClient(app) as c:  # lifespan trains the model because none exists yet
        yield c
