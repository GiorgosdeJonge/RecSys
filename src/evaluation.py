"""Offline evaluation utilities for Goodbooks recommendations.

This module implements a temporal holdout evaluation inspired by the
``simple_temporal_eval.py`` sketch. It splits each user's history so the last
interaction is held out for testing, fits a simple item–item model that relies
on historical ratings, and reports Hit@K, MRR@K, and coverage. A sanity check
re-runs scoring with user histories wiped to demonstrate dependence on past
ratings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED = ["user_id", "item_id", "rating", "timestamp"]


def prepare_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Normalize Goodbooks ratings for temporal evaluation.

    The Goodbooks ``ratings.csv`` file ships with ``user_id``, ``book_id``, and
    ``rating`` columns but no timestamp. This helper renames ``book_id`` to
    ``item_id`` (expected by the evaluator) and synthesizes a monotonically
    increasing ``timestamp`` per user using the original row order. The
    generated timestamp is sufficient for ordering interactions in a
    last-item holdout split.
    """

    df = ratings.copy()
    if "item_id" not in df.columns:
        if "book_id" in df.columns:
            df = df.rename(columns={"book_id": "item_id"})
        else:
            raise ValueError("ratings must include an 'item_id' or 'book_id' column")

    if "timestamp" not in df.columns:
        # Preserve file order within each user as a proxy for recency
        df = df.reset_index(drop=True)
        df["timestamp"] = df.groupby("user_id").cumcount()

    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("Empty dataframe.")
    if not (
        np.issubdtype(df["timestamp"].dtype, np.number)
        or np.issubdtype(df["timestamp"].dtype, np.datetime64)
    ):
        raise ValueError("timestamp must be numeric or datetime64")


def temporal_last_holdout(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_id", "timestamp"]).copy()
    train_parts, test_parts = [], []
    for _, g in df.groupby("user_id", sort=False):
        if len(g) < 2:
            # keep all in train; no test for this user
            train_parts.append(g)
            continue
        train_parts.append(g.iloc[:-1])
        test_parts.append(g.iloc[-1:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train, test


class ItemItemRecommender:
    """
    Item–item cosine on mean-centered ratings.
    Why: demonstrates dependence on user's past ratings to score unseen items.
    """

    def __init__(self, k_sim: int = 50, shrink: float = 10.0, min_overlap: int = 1):
        self.k_sim = k_sim
        self.shrink = shrink
        self.min_overlap = min_overlap

    def fit(self, train: pd.DataFrame) -> "ItemItemRecommender":
        self.users = train["user_id"].unique().tolist()
        self.items = train["item_id"].unique().tolist()
        self.u2i = {u: idx for idx, u in enumerate(self.users)}
        self.i2i = {i: idx for idx, i in enumerate(self.items)}

        rows = train["user_id"].map(self.u2i).to_numpy()
        cols = train["item_id"].map(self.i2i).to_numpy()
        vals = train["rating"].astype(float).to_numpy()
        self.R = sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.users), len(self.items)))

        # row-mean center
        sums = self.R.sum(axis=1).A1
        cnts = (self.R != 0).sum(axis=1).A1
        self.user_mean = np.divide(sums, cnts, out=np.zeros_like(sums, dtype=float), where=cnts > 0)

        R_c = self.R.tolil(copy=True)
        for u in range(R_c.shape[0]):
            mu = self.user_mean[u]
            if mu != 0:
                R_c.data[u] = [v - mu for v in R_c.data[u]]
        R_c = R_c.tocsr()

        # cosine similarity with shrinkage & overlap
        sim = cosine_similarity(R_c.T, dense_output=False)  # item x item
        overlap = (self.R.T > 0).astype(int) @ (self.R > 0).astype(int)  # item co-counts
        overlap = overlap.tocsr()
        sim = sim.multiply(overlap / (overlap + self.shrink))

        # keep top-k per item
        self.S = self._topk(sim.tolil(), self.k_sim).tocsr()

        # training seen per user
        self.seen: Dict[object, set] = train.groupby("user_id")["item_id"].apply(set).to_dict()
        return self

    def recommend(self, user_id: object, k: int = 10) -> List[Tuple[object, float]]:
        if user_id not in self.u2i:
            return []
        uidx = self.u2i[user_id]
        # score = R_u * S  (aggregate past ratings onto similar items)
        r_u = self.R.getrow(uidx)
        scores = r_u @ self.S
        scores = scores.toarray().ravel()

        # re-add user's mean to approximate absolute scale (not needed for ranking, harmless)
        scores = scores + self.user_mean[uidx]

        # filter seen items
        seen = self.seen.get(user_id, set())
        if seen:
            seen_idx = [self.i2i[i] for i in seen if i in self.i2i]
            scores[np.array(seen_idx, dtype=int)] = -np.inf

        # top-k
        k_eff = min(k, len(scores))
        if k_eff == 0:
            return []
        top = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
        top = top[np.argsort(-scores[top])]
        return [(self._item_from_idx(j), float(scores[j])) for j in top if np.isfinite(scores[j])]

    def _item_from_idx(self, j: int) -> object:
        # reverse index
        return self.items[j]

    @staticmethod
    def _topk(mat: sparse.lil_matrix, k: int) -> sparse.lil_matrix:
        for i in range(mat.shape[0]):
            row = mat.data[i]
            idx = mat.rows[i]
            if len(row) > k:
                sel = np.argpartition(np.array(row), -k)[-k:]
                mat.data[i] = list(np.array(row)[sel])
                mat.rows[i] = list(np.array(idx)[sel])
        return mat


@dataclass
class EvalSummary:
    users_evaluated: int
    hit_at_k: float
    mrr_at_k: float
    coverage: float  # fraction of users with >=1 recommendation


def evaluate_last_item(
    model: ItemItemRecommender, train: pd.DataFrame, test: pd.DataFrame, k: int = 10
) -> EvalSummary:
    # Per-user last item
    test_last = test.sort_values(["user_id", "timestamp"]).groupby("user_id").tail(1)
    users = test_last["user_id"].unique().tolist()

    hits, rr, have_recs = [], [], []
    for u in users:
        target_row = test_last[test_last["user_id"] == u].iloc[0]
        target_item = target_row["item_id"]
        recs = model.recommend(u, k=k)
        rec_items = [i for i, _ in recs]
        have_recs.append(1 if len(rec_items) > 0 else 0)

        if target_item in rec_items:
            rank = rec_items.index(target_item) + 1
            hits.append(1.0)
            rr.append(1.0 / rank)
        else:
            hits.append(0.0)
            rr.append(0.0)

    users_eval = len(users)
    hitk = float(np.mean(hits)) if users_eval else float("nan")
    mrrk = float(np.mean(rr)) if users_eval else float("nan")
    cov = float(np.mean(have_recs)) if users_eval else float("nan")
    return EvalSummary(users_eval, hitk, mrrk, cov)


def sanity_no_history_hit_mrr(train: pd.DataFrame, test: pd.DataFrame, k: int = 10) -> EvalSummary:
    """
    Refit model but pretend users have no past ratings by clearing the user rows in R.
    Expect Hit@k & MRR@k to collapse toward 0 if recommendations truly depend on past ratings.
    """

    model = ItemItemRecommender()
    model.fit(train)

    # wipe user histories at scoring time by monkey-patching R rows to zeros
    model_zero = ItemItemRecommender()
    model_zero.__dict__ = dict(model.__dict__)
    model_zero.R = sparse.csr_matrix(model.R.shape, dtype=model.R.dtype)

    return evaluate_last_item(model_zero, train, test, k=k)


def run_temporal_evaluation(ratings: pd.DataFrame, k: int = 10) -> Tuple[EvalSummary, EvalSummary]:
    """Perform temporal last-item evaluation and a no-history sanity check."""

    validate(ratings)
    train, test = temporal_last_holdout(ratings)

    if test.empty:
        raise ValueError("Not enough history per user for temporal holdout; need >=2 interactions each.")

    model = ItemItemRecommender()
    model.fit(train)

    eval_real = evaluate_last_item(model, train, test, k=k)
    eval_zero = sanity_no_history_hit_mrr(train, test, k=k)
    return eval_real, eval_zero


def format_eval_summary(summary: EvalSummary, k: int) -> List[str]:
    return [
        f"Users evaluated: {summary.users_evaluated}",
        f"Hit@{k}: {summary.hit_at_k:.4f}",
        f"MRR@{k}: {summary.mrr_at_k:.4f}",
        f"Coverage: {summary.coverage:.4f}",
    ]
