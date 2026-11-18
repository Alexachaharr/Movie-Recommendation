# Personalised Movie Recommendation System

A deployable hybrid movie recommender built with **Python, SQL and machine learning**. It serves
personalised recommendations to **10,000+ user profiles** from their rating behaviour. Inference
latency is kept low by SQL query tuning (a covering index, narrow queries, no N+1) and a
recommendation cache (in-process LRU or Redis).

- **FastAPI** REST API with OpenAPI docs at `/docs`
- **Hybrid model**: implicit CF (PureSVD) + explicit CF (biases + SVD) + content-based (TF-IDF) + popularity prior
- **Real-time personalisation**: new ratings are folded into the model on the fly, with no retraining
- **SQLite** (zero config) or **PostgreSQL**; in-memory cache or **Redis**
- Web UI, offline evaluation, latency benchmark, test suite, Docker / Compose / Render config

---

## Quick start (local, about 1 minute)

```bash
pip install -r requirements-dev.txt
python -m app.cli bootstrap          # creates schema, seeds 10k users / 2.5k movies / ~600k ratings, trains
uvicorn app.main:app --port 8000     # open http://localhost:8000
```

## Deploy

### Option A: Docker (any VPS, AWS/GCP/Azure VM, Fly.io, Railway)
```bash
docker build -t movie-recommender .
docker run -p 8000:8000 -e ADMIN_TOKEN=change-me movie-recommender
```
On first boot the container seeds the database and trains the model, then starts gunicorn with
uvicorn workers. That runs on SQLite inside the container, which is fine for a demo. **For real
deployments, set `DATABASE_URL` to Postgres**, because container disks are usually ephemeral.

### Option B: Docker Compose (app + PostgreSQL + Redis)
```bash
ADMIN_TOKEN=change-me docker compose up --build    # http://localhost:8000
```

### Option C: Render (managed, from GitHub)
1. Push this folder to a GitHub repo.
2. In Render: **New → Blueprint**, then pick the repo. `render.yaml` creates the web service, a Postgres
   database, and a disk for the model file. `ADMIN_TOKEN` is auto-generated.
3. The first deploy takes a few minutes while it seeds and trains.
   *(Check the plan names in `render.yaml` against Render's current pricing.)*

Any platform that runs a Dockerfile works the same way: set `DATABASE_URL`, optionally
`REDIS_URL`, and expose `$PORT`.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/movies.db` | SQLite or Postgres (`postgres://…` URLs are normalised automatically) |
| `REDIS_URL` | *(unset)* | Shared cache across workers/instances; falls back to in-memory if unreachable |
| `ADMIN_TOKEN` | *(unset)* | Enables `POST /api/admin/retrain` (send header `X-Admin-Token`) |
| `CACHE_TTL_SECONDS` | `600` | Recommendation cache TTL |
| `N_FACTORS` / `N_IMPLICIT` | `32` / `16` | SVD ranks for explicit / implicit CF |
| `CF_SHRINK` | `15` | Ratings needed before CF outweighs content/popularity |
| `SEED_USERS` / `SEED_MOVIES` | `10000` / `2500` | Size of the synthetic dataset on first boot |
| `WEB_CONCURRENCY` | `2` | gunicorn workers |

## Using real MovieLens data

Download a dataset from <https://grouplens.org/datasets/movielens/> (for example `ml-25m` or `ml-1m`), then:
```bash
python -m app.cli import-movielens ./ml-25m      # supports movies.csv/ratings.csv and ml-1m .dat files
python -m app.cli train
python -m app.cli evaluate
```
For ml-25m, use Postgres and tune `N_FACTORS` / `N_IMPLICIT` (try 64–128) based on `evaluate`.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/users/{id}/recommendations?n=10` | Top-N recommendations with signal breakdown and a "because you liked" explanation |
| GET | `/api/users/{id}/ratings` | Rating history |
| PUT | `/api/users/{id}/ratings/{movie_id}` | `{"rating": 4.5}` adds or updates a rating (0.5–5, steps of 0.5) and invalidates that user's cache |
| DELETE | `/api/users/{id}/ratings/{movie_id}` | Remove a rating |
| POST | `/api/users` | `{"name": "alice"}` creates a user (cold start → popularity, then content, then CF) |
| GET | `/api/movies/search?q=` | Title search |
| GET | `/api/movies/{id}` / `/similar` | Movie details / item-to-item similar movies |
| GET | `/api/stats`, `/api/metrics`, `/health` | DB/model/cache stats, per-route p50/p95 latency, health check |
| POST | `/api/admin/retrain` | Retrain from the DB (needs `ADMIN_TOKEN`); other workers hot-reload the new model |

## How it works

```
request ─► cache lookup (key = model version + user generation + n) ──hit──► response
               │ miss
               ▼
   SELECT movie_id, rating FROM ratings WHERE user_id = ?   ← served from the covering index
               ▼
   fold user into model:  implicit CF  +  explicit CF  +  content profile  +  popularity
               ▼
   blend: w_cf = n/(n+15) · CF  +  (1−w_cf) · (0.65 content + 0.35 popularity)
               ▼
   top-N, explanation, metadata from the in-memory catalog ─► cache ─► response
```

- **Implicit CF**: truncated SVD of the binary user×movie matrix. This captures "people who watched X also watched Y".
- **Explicit CF**: regularised user/item biases plus SVD of the residuals. This predicts the star rating.
- **Content**: TF-IDF over genres and decade; the user profile is the mean-centred, rating-weighted sum of rated movies.
- **Cold start**: a new user gets popularity (Bayesian-average quality × volume). As ratings arrive, the blend
  shifts to content and then to CF automatically.
- **Cache invalidation**: per-user generation counters. A new rating bumps the counter, and retraining changes
  the model version, so stale entries are never served. No key scans are needed, and it works across workers with Redis.
- **Model hot-reload**: models are saved atomically (`os.replace`). Every worker checks the file's mtime and reloads.

## Results (measured in this repo on the bundled synthetic dataset)

**Offline evaluation**: `python -m app.cli evaluate` (temporal hold-out of each user's last 20% of ratings, 2,000 users)

| Strategy | Precision@10 |
|---|---|
| Popularity | 0.029 |
| Content only | 0.007 |
| Implicit CF | 0.036 |
| **Hybrid** | **0.039** (+35% vs popularity) |

Rating RMSE: **0.659** for hybrid CF vs 0.722 for the item-mean baseline.

**Inference latency**: `python -m app.cli benchmark` (3,000 requests, Zipf-distributed users, ~50% cache hit rate)

| Path | PostgreSQL mean | SQLite mean |
|---|---|---|
| Untuned (`SELECT *`, N+1 metadata queries, no cache) | 5.0 ms | 1.9 ms |
| Tuned SQL (covering index, one narrow query, in-memory catalog) | 2.1 ms (−58%) | 1.6 ms (−15%) |
| Tuned SQL + cache | 0.9 ms (−81%) | 0.9 ms (−54%) |

The exact numbers depend on hardware, network round-trip to the database, and traffic. Rerun the benchmark
on your own deployment and quote those figures.

## Tests
```bash
pytest -q
```

## Project layout
```
app/
  main.py        FastAPI app, routes, latency middleware
  service.py     use cases: recommend, rate, similar, search, retrain
  model.py       hybrid recommender (train / save / load / fold-in scoring)
  db.py          schema, indexes, tuned queries, SQLite/Postgres specifics
  cache.py       TTL+LRU in-memory cache and Redis cache
  seed.py        synthetic 10k-user dataset and MovieLens importer
  evaluate.py    offline metrics      benchmark.py   latency benchmark
  cli.py         bootstrap | seed | import-movielens | train | evaluate | benchmark
  static/        web UI
tests/           API and model tests
Dockerfile, docker-compose.yml, render.yaml, docker/entrypoint.sh
```

## Production notes
- Retraining: run `python -m app.cli train` from a cron job or scheduled task (e.g. nightly), or call the admin endpoint.
  Day-to-day ratings still show up immediately through fold-in.
- There is no end-user authentication. Put the API behind your auth/gateway before exposing write endpoints publicly.
- Movies added after training appear in recommendations after the next retrain.
