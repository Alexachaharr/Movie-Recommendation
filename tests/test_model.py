import numpy as np

from app import db
from app.evaluate import evaluate
from app.model import HybridModel


def _data(client):  # depends on the seeded session database
    with db.get_engine().connect() as conn:
        return db.load_catalog(conn), db.load_all_ratings(conn)


def test_save_load_roundtrip(client, tmp_path):
    catalog, (u, m, r, _) = _data(client)
    model = HybridModel.train(u, m, r, catalog, n_factors=8, n_implicit=8)
    model.save(tmp_path / "m.npz")
    loaded = HybridModel.load(tmp_path / "m.npz")
    sel = u == u[0]
    a = model.recommend(m[sel], r[sel], n=5)["items"]
    b = loaded.recommend(m[sel], r[sel], n=5)["items"]
    assert [x["movie_id"] for x in a] == [x["movie_id"] for x in b]


def test_hybrid_beats_baselines(client):
    catalog, (u, m, r, ts) = _data(client)
    res = evaluate(u, m, r, ts, catalog, n_factors=16, n_implicit=8, max_users=400)
    assert res["rmse"]["cf_hybrid_biases"] < res["rmse"]["baseline_item_mean"]
    rk = res["ranking"]
    assert rk["hybrid"]["ndcg"] > rk["content"]["ndcg"]
    # random top-10 out of ~300 items would score roughly 10/300 precision at best
    assert rk["hybrid"]["precision"] > 0.03


def test_unknown_movies_ignored():
    catalog = {i: {"id": i, "title": f"m{i}", "year": 2000, "genres": "Drama"} for i in range(1, 21)}
    rng = np.random.default_rng(0)
    u = rng.integers(1, 30, 400)
    m = rng.integers(1, 21, 400)
    r = rng.choice([1.0, 3.0, 5.0], 400)
    model = HybridModel.train(u, m, r, catalog, n_factors=4, n_implicit=4)
    out = model.recommend([999, 1000], [5.0, 4.0], n=3)
    assert out["n_ratings_used"] == 0 and len(out["items"]) == 3
