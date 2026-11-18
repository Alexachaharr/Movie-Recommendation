"""Offline evaluation with a temporal per-user hold-out.

For each user with >= 5 ratings, their most recent 20% of ratings are held out. The model is
trained on the rest. We report:
  * RMSE of predicted ratings (CF fold-in) vs. the bias-only baseline (mu + b_i)
  * Precision@K, Recall@K, NDCG@K (relevant = held-out rating >= 4.0) for
    popularity, content-only, CF-only and the hybrid blend.
"""
from __future__ import annotations

import time

import numpy as np

from .model import HybridModel, _z


def _ndcg(hits: np.ndarray, n_rel: int, k: int) -> float:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((hits * disc[: len(hits)]).sum())
    idcg = float(disc[: min(n_rel, k)].sum())
    return dcg / idcg if idcg else 0.0


def evaluate(user_ids, movie_ids, values, ts, catalog, *, k=10, test_frac=0.2, max_users=2000,
             seed=0, **train_kwargs) -> dict:
    t0 = time.time()
    user_ids, movie_ids = np.asarray(user_ids), np.asarray(movie_ids)
    values, ts = np.asarray(values, dtype=np.float32), np.asarray(ts)
    order = np.lexsort((ts, user_ids))
    user_ids, movie_ids, values, ts = user_ids[order], movie_ids[order], values[order], ts[order]
    starts = np.r_[0, np.flatnonzero(np.diff(user_ids)) + 1]
    ends = np.r_[starts[1:], len(user_ids)]

    is_test = np.zeros(len(user_ids), dtype=bool)
    for s, e in zip(starts, ends):
        cnt = e - s
        if cnt >= 5:
            is_test[e - max(1, int(round(cnt * test_frac))): e] = True

    model = HybridModel.train(user_ids[~is_test], movie_ids[~is_test], values[~is_test], catalog, **train_kwargs)

    rng = np.random.default_rng(seed)
    eligible = [i for i, (s, e) in enumerate(zip(starts, ends)) if is_test[s:e].any()]
    sample = rng.choice(eligible, size=min(max_users, len(eligible)), replace=False)

    sq_cf, sq_base, n_pred = 0.0, 0.0, 0
    strategies = ("popularity", "content", "explicit_cf", "implicit_cf", "collaborative", "hybrid")
    agg = {s: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for s in strategies}
    n_rank = 0
    for ui in sample:
        s, e = starts[ui], ends[ui]
        tr, te = ~is_test[s:e], is_test[s:e]
        tr_m, tr_r = movie_ids[s:e][tr], values[s:e][tr]
        te_m, te_r = movie_ids[s:e][te], values[s:e][te]
        sc = model.score(tr_m, tr_r)
        te_idx = np.array([model.index[int(m)] for m in te_m if int(m) in model.index])
        te_val = np.array([r for m, r in zip(te_m, te_r) if int(m) in model.index])
        if len(te_idx) == 0:
            continue
        pred = np.clip(sc["cf_pred"][te_idx], 0.5, 5.0)
        base = np.clip(model.mu + model.item_bias[te_idx], 0.5, 5.0)
        sq_cf += float(((pred - te_val) ** 2).sum())
        sq_base += float(((base - te_val) ** 2).sum())
        n_pred += len(te_idx)

        relevant = set(te_idx[te_val >= 4.0].tolist())
        if not relevant:
            continue
        n_rank += 1
        scores = {
            "popularity": model.z_pop.astype(np.float64),
            "content": _z(sc["content"]).astype(np.float64),
            "explicit_cf": sc["cf_pred"].astype(np.float64),
            "implicit_cf": sc["implicit"].astype(np.float64),
            "collaborative": sc["cf"].astype(np.float64),
            "hybrid": sc["hybrid"].astype(np.float64),
        }
        for name, vec in scores.items():
            v = vec.copy()
            if len(sc["idx"]):
                v[sc["idx"]] = -np.inf
            top = np.argpartition(-v, k - 1)[:k]
            top = top[np.argsort(-v[top])]
            hits = np.array([1.0 if t in relevant else 0.0 for t in top])
            agg[name]["precision"] += hits.sum() / k
            agg[name]["recall"] += hits.sum() / len(relevant)
            agg[name]["ndcg"] += _ndcg(hits, len(relevant), k)

    ranking = {s: {m: round(float(v) / max(n_rank, 1), 4) for m, v in d.items()} for s, d in agg.items()}
    return {
        "k": k,
        "users_evaluated": int(n_rank),
        "rmse": {"cf_hybrid_biases": round((sq_cf / n_pred) ** 0.5, 4),
                 "baseline_item_mean": round((sq_base / n_pred) ** 0.5, 4)},
        "ranking": ranking,
        "seconds": round(time.time() - t0, 1),
    }
