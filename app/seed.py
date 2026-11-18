"""Load data into the database.

Two sources:
  * ``generate_synthetic`` - a realistic synthetic dataset (default 10,000 users) with latent
    taste structure, genre preferences, popularity skew and rating noise, so the recommender
    has real signal to learn. Fully offline and deterministic (seeded).
  * ``import_movielens`` - the real GroupLens MovieLens datasets (ml-latest-small, ml-25m,
    ml-32m as CSV, or ml-1m as .dat). Download from https://grouplens.org/datasets/movielens/
"""
from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import numpy as np
from sqlalchemy import delete
from sqlalchemy.engine import Engine

from . import db

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]

_ADJ = ["Silent", "Crimson", "Last", "Hidden", "Broken", "Golden", "Midnight", "Electric", "Lost",
        "Savage", "Quiet", "Endless", "Burning", "Frozen", "Secret", "Wild", "Distant", "Iron",
        "Velvet", "Hollow", "Bright", "Dark", "Little", "Final", "Restless", "Paper", "Glass",
        "Neon", "Sunken", "Northern", "Wandering", "Shattered", "Scarlet", "Faded", "Rising"]
_NOUN = ["Harbor", "Empire", "Garden", "Signal", "Kingdom", "Detective", "Horizon", "River",
         "Machine", "Promise", "Frontier", "Shadow", "Orchestra", "Witness", "Voyage", "Planet",
         "Summer", "Letter", "Circus", "Station", "Island", "Protocol", "Heist", "Ballad", "Storm",
         "Mirror", "Legacy", "Carnival", "Outpost", "Lighthouse", "Cartographer", "Paradox",
         "Wolves", "Orbit", "Requiem", "Alibi", "Monsoon", "Labyrinth", "Comet", "Vendetta"]
_PATTERNS = ["The {a} {n}", "{a} {n}", "The {a} {n} Chronicles", "A {a} {n}", "The {n}", "{a} {n} II",
             "Return of the {n}", "Beyond the {a} {n}", "Children of the {n}", "The {a} {n} Affair"]


def _clear(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(delete(db.ratings))
        conn.execute(delete(db.users))
        conn.execute(delete(db.movies))


def _bulk_insert(engine: Engine, table, rows: list[dict], chunk: int = 20_000) -> None:
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk):
            conn.execute(table.insert(), rows[i:i + chunk])


def generate_synthetic(engine: Engine, n_users: int = 10_000, n_movies: int = 2_500,
                       mean_ratings: int = 60, seed: int = 42, verbose: bool = True) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    G, K = len(GENRES), 16
    genre_emb = rng.normal(0, 1, (G, K))

    # ---- movies: 1-3 genres, latent vector anchored on genres, quality + popularity
    movie_rows, q, b_i = [], np.zeros((n_movies, K)), rng.normal(0, 0.35, n_movies)
    genre_pop = rng.dirichlet(np.ones(G) * 1.5)
    used = set()
    for m in range(n_movies):
        ng = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20])
        gs = sorted(rng.choice(G, size=ng, replace=False, p=genre_pop))
        q[m] = genre_emb[gs].mean(0) + rng.normal(0, 0.5, K)
        year = int(np.clip(rng.normal(2003, 14), 1950, 2025))
        while True:
            title = rng.choice(_PATTERNS).format(a=rng.choice(_ADJ), n=rng.choice(_NOUN))
            if (title, year) not in used:
                used.add((title, year))
                break
        movie_rows.append({"id": m + 1, "title": f"{title} ({year})", "year": year,
                           "genres": "|".join(GENRES[g] for g in gs)})
    q /= np.sqrt(K)
    popularity = rng.lognormal(0, 1.1, n_movies) * np.exp(b_i)  # better movies get watched more
    popularity /= popularity.sum()

    # ---- users: 2-4 favourite genres define taste; activity is long-tailed
    user_rows, rating_rows = [], []
    t_start, t_end = 1_420_070_400, 1_760_000_000  # 2015 .. 2025
    activity = np.clip(rng.lognormal(np.log(mean_ratings) - 0.35, 0.85, n_users), 5, min(600, n_movies // 2)).astype(int)
    for u in range(n_users):
        favs = rng.choice(G, size=rng.integers(2, 5), replace=False)
        p_u = (genre_emb[favs].mean(0) + rng.normal(0, 0.6, K)) / np.sqrt(K)
        b_u = rng.normal(0, 0.3)
        affinity = q @ p_u
        w = popularity * np.exp(1.8 * affinity)  # people mostly watch what they expect to like
        w /= w.sum()
        items = rng.choice(n_movies, size=activity[u], replace=False, p=w)
        raw = 3.55 + b_u + b_i[items] + 2.2 * affinity[items] + rng.normal(0, 0.55, len(items))
        vals = np.clip(np.round(raw * 2) / 2, 0.5, 5.0)
        ts = np.sort(rng.integers(t_start, t_end, len(items)))
        uid = u + 1
        user_rows.append({"id": uid, "name": f"user_{uid}", "created_at": int(ts[0])})
        rating_rows.extend({"user_id": uid, "movie_id": int(m) + 1, "rating": float(r), "ts": int(t)}
                           for m, r, t in zip(items, vals, ts))

    _clear(engine)
    _bulk_insert(engine, db.movies, movie_rows)
    _bulk_insert(engine, db.users, user_rows)
    _bulk_insert(engine, db.ratings, rating_rows)
    with engine.begin() as conn:
        db.reset_pg_sequences(conn)
    db.analyze(engine)
    stats = {"users": n_users, "movies": n_movies, "ratings": len(rating_rows),
             "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"[seed] synthetic dataset loaded: {stats}")
    return stats


_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def import_movielens(engine: Engine, path: str | Path, verbose: bool = True) -> dict:
    """Import a MovieLens folder (movies.csv + ratings.csv, or movies.dat + ratings.dat)."""
    t0 = time.time()
    path = Path(path)
    movie_rows, rating_rows = [], []
    if (path / "movies.csv").exists():
        with open(path / "movies.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = _YEAR_RE.search(row["title"])
                movie_rows.append({"id": int(row["movieId"]), "title": row["title"].strip(),
                                   "year": int(m.group(1)) if m else None, "genres": row["genres"]})
        with open(path / "ratings.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rating_rows.append({"user_id": int(row["userId"]), "movie_id": int(row["movieId"]),
                                    "rating": float(row["rating"]), "ts": int(row["timestamp"])})
    elif (path / "movies.dat").exists():  # ml-1m / ml-10m
        with open(path / "movies.dat", encoding="latin-1") as f:
            for line in f:
                mid, title, genres = line.rstrip("\n").split("::")
                m = _YEAR_RE.search(title)
                movie_rows.append({"id": int(mid), "title": title.strip(),
                                   "year": int(m.group(1)) if m else None, "genres": genres})
        with open(path / "ratings.dat", encoding="latin-1") as f:
            for line in f:
                uid, mid, r, ts = line.strip().split("::")
                rating_rows.append({"user_id": int(uid), "movie_id": int(mid), "rating": float(r), "ts": int(ts)})
    else:
        raise FileNotFoundError(f"No movies.csv or movies.dat in {path}")

    user_first: dict[int, int] = {}
    for r in rating_rows:
        u = r["user_id"]
        if u not in user_first or r["ts"] < user_first[u]:
            user_first[u] = r["ts"]
    user_rows = [{"id": u, "name": f"user_{u}", "created_at": t} for u, t in sorted(user_first.items())]

    _clear(engine)
    _bulk_insert(engine, db.movies, movie_rows)
    _bulk_insert(engine, db.users, user_rows)
    _bulk_insert(engine, db.ratings, rating_rows, chunk=50_000)
    with engine.begin() as conn:
        db.reset_pg_sequences(conn)
    db.analyze(engine)
    stats = {"users": len(user_rows), "movies": len(movie_rows), "ratings": len(rating_rows),
             "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"[seed] MovieLens loaded from {path}: {stats}")
    return stats
