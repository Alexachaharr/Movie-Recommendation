"""Hybrid recommender: collaborative filtering + content-based + popularity prior.

Collaborative part (rating behaviour) - two complementary signals
  * Implicit ("who watched what"): truncated SVD of the binary user x item matrix
    (PureSVD). Captures co-consumption patterns; strongest for top-N ranking.
        implicit_ui = W_i . sum_{j in rated(u)} W_j
  * Explicit ("how much they liked it"): baseline predictor r_ui ~ mu + b_u + b_i
    (regularised biases) plus a truncated SVD of the residual matrix:
        u = resid_u @ V[rated]      ->     cf_ui = mu + b_u + b_i + V_i . u
  Both are *folded in* at request time, so a brand-new rating changes recommendations
  immediately without retraining.  CF = 0.5 * z(implicit) + 0.5 * z(explicit).

Content part
    TF-IDF over genre and decade tokens. A user profile is the rating-weighted
    (mean-centred) sum of the item vectors they rated; items are scored by cosine.

Blend
    Each signal is z-normalised, then mixed with a confidence weight that grows with the
    number of ratings n:  w_cf = n / (n + CF_SHRINK).  Cold users lean on content and
    popularity; users with many ratings are driven by CF. Zero ratings -> popularity.
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _z(x: np.ndarray) -> np.ndarray:
    s = float(x.std())
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def _doc(genres: str, year: int | None) -> str:
    toks = [
        "g_" + g.strip().lower().replace(" ", "_").replace("-", "_")
        for g in (genres or "").split("|")
        if g.strip() and g.strip() != "(no genres listed)"
    ]
    if year:
        toks.append(f"decade_{(int(year) // 10) * 10}")
    return " ".join(toks) or "unknown"


class HybridModel:
    CONTENT_SHARE = 0.65   # of the non-CF weight, how much goes to content (rest = popularity)

    def __init__(self, movie_ids, mu, item_bias, V, W, content, popularity, rating_count, meta):
        self.movie_ids = np.asarray(movie_ids, dtype=np.int64)
        self.index = {int(m): i for i, m in enumerate(self.movie_ids)}
        self.mu = float(mu)
        self.item_bias = np.asarray(item_bias, dtype=np.float32)
        self.V = np.asarray(V, dtype=np.float32)
        self.W = np.asarray(W, dtype=np.float32)
        self.content: sp.csr_matrix = content.tocsr().astype(np.float32)
        self.popularity = np.asarray(popularity, dtype=np.float32)
        self.rating_count = np.asarray(rating_count, dtype=np.int32)
        self.meta = meta
        self.version: str = meta["version"]
        self.reg_user = float(meta["params"]["reg_user"])
        self.cf_shrink = float(meta["params"]["cf_shrink"])
        self.z_pop = _z(self.popularity).astype(np.float32)
        self.V_norm = normalize(self.V).astype(np.float32)
        self.W_norm = normalize(self.W).astype(np.float32)
        # confidence of an item's CF vector (few ratings -> noisy vector)
        self.cf_conf = (self.rating_count / (self.rating_count + 20.0)).astype(np.float32)

    @property
    def n_items(self) -> int:
        return len(self.movie_ids)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, user_ids, movie_ids, values, catalog: dict[int, dict], *, n_factors=32, n_implicit=16,
              reg_item=25.0, reg_user=10.0, cf_shrink=15.0, seed=0) -> "HybridModel":
        t0 = time.time()
        item_ids = np.array(sorted(catalog), dtype=np.int64)
        index = {int(m): i for i, m in enumerate(item_ids)}
        keep = np.fromiter((int(m) in index for m in movie_ids), dtype=bool, count=len(movie_ids))
        user_ids, movie_ids, values = np.asarray(user_ids)[keep], np.asarray(movie_ids)[keep], np.asarray(values, dtype=np.float64)[keep]
        it = np.fromiter((index[int(m)] for m in movie_ids), dtype=np.int64, count=len(movie_ids))
        _, us = np.unique(user_ids, return_inverse=True)
        n_users, n_items = int(us.max()) + 1, len(item_ids)

        mu = float(values.mean())
        n_i = np.bincount(it, minlength=n_items).astype(np.float64)
        b_i = np.bincount(it, weights=values - mu, minlength=n_items) / (n_i + reg_item)
        n_u = np.bincount(us, minlength=n_users).astype(np.float64)
        b_u = np.bincount(us, weights=values - mu - b_i[it], minlength=n_users) / (n_u + reg_user)
        resid = values - mu - b_i[it] - b_u[us]

        R = sp.csr_matrix((resid, (us, it)), shape=(n_users, n_items))
        k = max(1, min(n_factors, min(R.shape) - 1))
        rs = np.random.default_rng(seed)
        _, s, Vt = svds(R, k=k, v0=rs.uniform(-1, 1, min(R.shape)))
        V = Vt.T[:, np.argsort(-s)]

        X = sp.csr_matrix((np.ones(len(it)), (us, it)), shape=(n_users, n_items))
        k2 = max(1, min(n_implicit, min(X.shape) - 1))
        _, s2, Wt = svds(X, k=k2, v0=rs.uniform(-1, 1, min(X.shape)))
        W = Wt.T[:, np.argsort(-s2)]

        docs = [_doc(catalog[int(m)]["genres"], catalog[int(m)]["year"]) for m in item_ids]
        content = normalize(TfidfVectorizer(token_pattern=r"\S+", lowercase=False).fit_transform(docs))

        # popularity prior: Bayesian-average quality + log-volume
        popularity = _z(mu + b_i) + 0.5 * _z(np.log1p(n_i))

        meta = {
            "version": time.strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "trained_at": int(time.time()),
            "train_seconds": round(time.time() - t0, 2),
            "n_users": n_users, "n_items": n_items, "n_ratings": int(len(values)),
            "params": {"n_factors": k, "n_implicit": k2, "reg_item": reg_item, "reg_user": reg_user, "cf_shrink": cf_shrink},
        }
        return cls(item_ids, mu, b_i, V, W, content, popularity, n_i, meta)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        c = self.content
        np.savez_compressed(
            buf, movie_ids=self.movie_ids, item_bias=self.item_bias, V=self.V, W=self.W,
            popularity=self.popularity, rating_count=self.rating_count,
            c_data=c.data, c_indices=c.indices, c_indptr=c.indptr, c_shape=np.array(c.shape),
            mu=np.array(self.mu), meta=np.frombuffer(json.dumps(self.meta).encode(), dtype=np.uint8),
        )
        tmp = path.with_suffix(f".tmp-{uuid.uuid4().hex[:6]}")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, path)  # atomic swap: running workers never see a half-written file

    @classmethod
    def load(cls, path: str | Path) -> "HybridModel":
        with np.load(path) as z:
            content = sp.csr_matrix((z["c_data"], z["c_indices"], z["c_indptr"]), shape=tuple(z["c_shape"]))
            meta = json.loads(z["meta"].tobytes().decode())
            return cls(z["movie_ids"], float(z["mu"]), z["item_bias"], z["V"], z["W"], content,
                       z["popularity"], z["rating_count"], meta)

    # ------------------------------------------------------------------ inference
    def _known(self, rated_ids, rated_vals):
        pairs = [(self.index[int(m)], float(r)) for m, r in zip(rated_ids, rated_vals) if int(m) in self.index]
        if not pairs:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        idx, r = zip(*pairs)
        return np.asarray(idx, dtype=np.int64), np.asarray(r, dtype=np.float32)

    def score(self, rated_ids, rated_vals) -> dict:
        """Score every item for a user given their (movie_id, rating) history."""
        idx, r = self._known(rated_ids, rated_vals)
        n = len(idx)
        base = self.mu + self.item_bias
        if n == 0:
            zeros = np.zeros(self.n_items, np.float32)
            return {"idx": idx, "r": r, "n": 0, "w_cf": 0.0, "cf_pred": base.copy(), "implicit": zeros,
                    "cf": zeros,
                    "content": np.zeros(self.n_items, np.float32), "hybrid": self.z_pop.copy()}

        b_u = float((r - self.mu - self.item_bias[idx]).sum() / (n + self.reg_user))
        resid = r - self.mu - self.item_bias[idx] - b_u
        u = resid @ self.V[idx]
        cf_pred = base + b_u + self.V @ u
        implicit = self.W @ self.W[idx].sum(0)
        cf = 0.5 * _z(implicit) + 0.5 * _z(cf_pred)

        w = r - r.mean()
        if np.allclose(w, 0):
            w = r / 5.0
        profile = np.asarray(self.content[idx].T @ w).ravel()
        norm = np.linalg.norm(profile)
        cb = (self.content @ profile) / norm if norm > 0 else np.zeros(self.n_items, np.float32)

        w_cf = n / (n + self.cf_shrink)
        hybrid = (w_cf * cf
                  + (1 - w_cf) * (self.CONTENT_SHARE * _z(cb) + (1 - self.CONTENT_SHARE) * self.z_pop))
        return {"idx": idx, "r": r, "n": n, "w_cf": w_cf, "cf_pred": cf_pred,
                "implicit": implicit, "cf": cf, "content": np.asarray(cb, dtype=np.float32), "hybrid": hybrid}

    def recommend(self, rated_ids, rated_vals, n: int = 10, exclude_rated: bool = True) -> dict:
        s = self.score(rated_ids, rated_vals)
        h = s["hybrid"].astype(np.float64).copy()
        if exclude_rated and len(s["idx"]):
            h[s["idx"]] = -np.inf
        n = max(0, min(n, int(np.isfinite(h).sum())))
        if n == 0:
            return {"items": [], "n_ratings_used": s["n"], "cf_weight": round(s["w_cf"], 3)}
        top = np.argpartition(-h, n - 1)[:n]
        top = top[np.argsort(-h[top])]

        # explanation: most similar highly-rated movie from the user's history
        because = [None] * len(top)
        if s["n"]:
            idx, r = s["idx"], s["r"]
            liked = idx[r >= max(3.5, float(r.mean()))]
            if len(liked) == 0:
                liked = idx[[int(np.argmax(r))]]
            liked = liked[:200]
            sims = (0.4 * (self.W_norm[top] @ self.W_norm[liked].T)
                    + 0.3 * (self.V_norm[top] @ self.V_norm[liked].T)) * self.cf_conf[top][:, None] \
                + 0.3 * (self.content[top] @ self.content[liked].T).toarray()
            best = sims.argmax(1)
            because = [int(self.movie_ids[liked[b]]) for b in best]

        items = []
        for rank, (i, b) in enumerate(zip(top, because), 1):
            items.append({
                "movie_id": int(self.movie_ids[i]),
                "rank": rank,
                "score": round(float(h[i]), 4),
                "predicted_rating": round(float(np.clip(s["cf_pred"][i], 0.5, 5.0)), 2),
                "signals": {
                    "collaborative": round(float(s["cf"][i]), 3),
                    "content": round(float(s["content"][i]), 3),
                    "popularity": round(float(self.z_pop[i]), 3),
                },
                "because_you_liked": b,
            })
        return {"items": items, "n_ratings_used": s["n"], "cf_weight": round(s["w_cf"], 3)}

    def similar(self, movie_id: int, n: int = 10) -> list[dict]:
        if int(movie_id) not in self.index:
            return []
        i = self.index[int(movie_id)]
        conf = self.cf_conf * self.cf_conf[i]
        sims = (0.4 * (self.W_norm @ self.W_norm[i]) + 0.3 * (self.V_norm @ self.V_norm[i])) * conf \
            + 0.3 * np.asarray((self.content @ self.content[i].T).todense()).ravel()
        sims[i] = -np.inf
        n = min(n, self.n_items - 1)
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top])]
        return [{"movie_id": int(self.movie_ids[j]), "similarity": round(float(sims[j]), 4)} for j in top]
