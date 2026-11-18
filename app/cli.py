"""Command-line entry point.

    python -m app.cli bootstrap                 # create schema, seed if empty, train if no model
    python -m app.cli seed --users 10000        # (re)generate the synthetic dataset
    python -m app.cli import-movielens ./ml-25m # load real MovieLens data instead
    python -m app.cli train                     # retrain and save the model
    python -m app.cli evaluate                  # offline metrics (RMSE, P@K, R@K, NDCG@K)
    python -m app.cli benchmark                 # latency: untuned vs tuned SQL vs tuned + cache
"""
from __future__ import annotations

import argparse
import json

from . import config, db, seed


def _engine():
    engine = db.get_engine()
    db.init_db(engine)
    return engine


def cmd_seed(a):
    seed.generate_synthetic(_engine(), n_users=a.users, n_movies=a.movies, mean_ratings=a.mean_ratings, seed=a.seed)


def cmd_import(a):
    seed.import_movielens(_engine(), a.path)


def cmd_train(_a):
    from .service import RecommenderService
    meta = RecommenderService(_engine()).train()
    print(f"[train] saved {config.MODEL_PATH}\n{json.dumps(meta, indent=2)}")


def cmd_bootstrap(_a):
    engine = _engine()
    with engine.connect() as conn:
        c = db.counts(conn)
    if c["ratings"] == 0:
        print("[bootstrap] empty database - generating synthetic dataset")
        seed.generate_synthetic(engine, config.SEED_USERS, config.SEED_MOVIES, config.SEED_MEAN_RATINGS)
    else:
        print(f"[bootstrap] database ready: {c}")
    if not config.MODEL_PATH.exists():
        cmd_train(_a)
    else:
        print(f"[bootstrap] model present at {config.MODEL_PATH}")


def cmd_evaluate(a):
    from .evaluate import evaluate
    engine = _engine()
    with engine.connect() as conn:
        catalog = db.load_catalog(conn)
        u, m, r, ts = db.load_all_ratings(conn)
    res = evaluate(u, m, r, ts, catalog, k=a.k, max_users=a.max_users, n_factors=config.N_FACTORS,
                   n_implicit=config.N_IMPLICIT, reg_item=config.REG_ITEM, reg_user=config.REG_USER,
                   cf_shrink=config.CF_SHRINK)
    print(json.dumps(res, indent=2))


def cmd_benchmark(a):
    from .benchmark import run
    print(json.dumps(run(_engine(), requests=a.requests, n=a.n), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--users", type=int, default=config.SEED_USERS)
    s.add_argument("--movies", type=int, default=config.SEED_MOVIES)
    s.add_argument("--mean-ratings", type=int, default=config.SEED_MEAN_RATINGS)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_seed)
    s = sub.add_parser("import-movielens")
    s.add_argument("path")
    s.set_defaults(fn=cmd_import)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    sub.add_parser("bootstrap").set_defaults(fn=cmd_bootstrap)
    s = sub.add_parser("evaluate")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--max-users", type=int, default=2000)
    s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("benchmark")
    s.add_argument("--requests", type=int, default=3000)
    s.add_argument("--n", type=int, default=10)
    s.set_defaults(fn=cmd_benchmark)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
