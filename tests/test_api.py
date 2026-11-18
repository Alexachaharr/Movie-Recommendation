def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_recommendations_exclude_rated_and_are_sorted(client):
    hist = client.get("/api/users/7/ratings", params={"limit": 1000}).json()
    rated = {x["id"] for x in hist["ratings"]}
    d = client.get("/api/users/7/recommendations", params={"n": 15}).json()
    ids = [i["movie_id"] for i in d["items"]]
    assert len(ids) == 15 and len(set(ids)) == 15
    assert not rated & set(ids)
    scores = [i["score"] for i in d["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(i["title"] for i in d["items"])


def test_cache_hit_then_invalidated_by_new_rating(client):
    first = client.get("/api/users/11/recommendations").json()
    second = client.get("/api/users/11/recommendations").json()
    assert second["cached"] is True
    top = first["items"][0]["movie_id"]
    assert client.put(f"/api/users/11/ratings/{top}", json={"rating": 0.5}).status_code == 200
    third = client.get("/api/users/11/recommendations").json()
    assert third["cached"] is False
    assert top not in [i["movie_id"] for i in third["items"]]


def test_cold_start_then_personalised(client):
    uid = client.post("/api/users", json={"name": "newbie"}).json()["id"]
    cold = client.get(f"/api/users/{uid}/recommendations").json()
    assert cold["n_ratings_used"] == 0 and "popularity" in cold["strategy"]
    for m in (3, 4, 5):
        client.put(f"/api/users/{uid}/ratings/{m}", json={"rating": 5})
    warm = client.get(f"/api/users/{uid}/recommendations").json()
    assert warm["n_ratings_used"] == 3 and 0 < warm["cf_weight"] < 0.5
    assert all(i["because_you_liked"]["id"] in (3, 4, 5) for i in warm["items"])


def test_similar_and_search(client):
    s = client.get("/api/movies/10/similar", params={"n": 5}).json()
    assert len(s["similar"]) == 5 and 10 not in [x["movie_id"] for x in s["similar"]]
    title_word = client.get("/api/movies/10").json()["title"].split()[1].lower()
    res = client.get("/api/movies/search", params={"q": title_word}).json()["results"]
    assert any(title_word in m["title"].lower() for m in res)


def test_validation_and_errors(client):
    assert client.put("/api/users/1/ratings/1", json={"rating": 4.3}).status_code == 422
    assert client.put("/api/users/1/ratings/1", json={"rating": 6}).status_code == 422
    assert client.put("/api/users/1/ratings/99999", json={"rating": 4}).status_code == 404
    assert client.get("/api/users/999999/recommendations").status_code == 404
    assert client.delete("/api/users/1/ratings/99999").status_code == 404


def test_admin_retrain_requires_token(client):
    assert client.post("/api/admin/retrain").status_code == 401
    r = client.post("/api/admin/retrain", headers={"X-Admin-Token": "test-token"})
    assert r.status_code == 200 and r.json()["n_ratings"] > 0
