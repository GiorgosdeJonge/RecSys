"""CLI for a content-based Goodbooks-10k recommender."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .data_loader import (
    TagMatrices,
    build_tag_matrix,
    load_book_tags,
    load_books,
    load_ratings,
    load_tags,
)
from .evaluation import format_eval_summary, prepare_ratings, run_temporal_evaluation
from .feature_builder import build_feature_matrix


@dataclass(frozen=True)
class UserRatingProfile:
    """Captures a normalized user vector and the rated books that informed it."""

    vector: sparse.csr_matrix
    rated_books: List[Dict[str, object]]


class ContentBasedRecommender:
    """Content-based recommender supporting title and tag-preference flows."""

    def __init__(self, feature_matrix, books):
        self.feature_matrix = feature_matrix
        self.books = books

    def _find_seed_indices(self, title_query: str) -> List[int]:
        mask = self.books["title"].str.contains(title_query, case=False, na=False) | self.books[
            "original_title"
        ].str.contains(title_query, case=False, na=False)
        return mask[mask].index.tolist()

    def recommend_from_title(self, title_query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Return (title, author, similarity) for the top recommendations."""

        seed_indices = self._find_seed_indices(title_query)
        if not seed_indices:
            raise ValueError(f"No books match the query '{title_query}'. Try a different keyword.")

        seed_matrix = self.feature_matrix[seed_indices]
        scores = cosine_similarity(seed_matrix, self.feature_matrix).mean(axis=0)

        # Exclude the seeds themselves by setting their scores to -inf
        scores[seed_indices] = -np.inf
        top_indices = np.argsort(scores)[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            row = self.books.iloc[idx]
            recommendations.append((row.get("title") or row.get("original_title"), row.get("authors", ""), scores[idx]))
        return recommendations


class TagPreferenceRecommender:
    """Recommend books whose tags align with provided user preferences."""

    def __init__(self, tag_matrices: TagMatrices, books: pd.DataFrame, tags: pd.DataFrame):
        self.tag_matrices = tag_matrices
        self.books_lookup = books.set_index("goodreads_book_id")
        self.tag_lookup = tags.set_index("tag_id")["tag_name"] if "tag_name" in tags.columns else None
        self.idx_to_tag_id = {idx: tag_id for tag_id, idx in tag_matrices.tag_id_to_idx.items()}
        self.book_id_to_goodreads = (
            books.set_index("book_id")["goodreads_book_id"].to_dict() if "book_id" in books.columns else {}
        )

        # L2-normalized TF–IDF transform of tag counts (item representation)
        transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
        self.tfidf_matrix = transformer.fit_transform(tag_matrices.matrix)

    def _make_user_vector(
        self,
        preferred_tags: Optional[Iterable[int]] = None,
        preferred_tag_weights: Optional[Dict[int, float]] = None,
    ) -> sparse.csr_matrix:
        if preferred_tag_weights is None and preferred_tags is None:
            raise ValueError("Provide preferred_tags or preferred_tag_weights to describe the user profile.")

        if preferred_tag_weights is None and preferred_tags is not None:
            preferred_tag_weights = {int(tag_id): 1.0 for tag_id in preferred_tags}

        indices, values = [], []
        for tag_id, weight in preferred_tag_weights.items():
            col = self.tag_matrices.tag_id_to_idx.get(int(tag_id))
            if col is not None:
                indices.append(col)
                values.append(float(weight))

        if not indices:
            raise ValueError("None of the provided tag_ids exist in the tag matrix. Check your inputs.")

        ind = np.zeros(len(indices), dtype=np.int32)
        cols = np.array(indices, dtype=np.int32)
        data = np.array(values, dtype=np.float32)
        user = sparse.csr_matrix((data, (ind, cols)), shape=(1, self.tag_matrices.matrix.shape[1]), dtype=np.float32)

        return normalize(user, norm="l2", axis=1, copy=False)

    def make_user_vector(
        self,
        *,
        preferred_tags: Optional[Iterable[int]] = None,
        preferred_tag_weights: Optional[Dict[int, float]] = None,
    ) -> sparse.csr_matrix:
        """Public helper to build an L2-normalized user preference vector."""

        return self._make_user_vector(preferred_tags, preferred_tag_weights)

    def build_user_vector_from_ratings(
        self,
        ratings: pd.DataFrame,
        user_id: int,
        *,
        min_rating: float = 4.0,
    ) -> sparse.csr_matrix:
        """Backward-compatible wrapper that returns only the vector."""

        profile = self.build_user_profile_from_ratings(ratings, user_id, min_rating=min_rating)
        return profile.vector

    def build_user_profile_from_ratings(
        self,
        ratings: pd.DataFrame,
        user_id: int,
        *,
        min_rating: float = 4.0,
    ) -> "UserRatingProfile":
        """Construct a preference vector and capture the rated books that inform it.

        Ratings are filtered by ``min_rating`` and weighted by the rating value.
        The resulting vector is normalized to live in the same space as items.
        """

        user_rows = ratings[ratings["user_id"] == user_id]
        if user_rows.empty:
            raise ValueError(f"No ratings found for user_id {user_id}.")

        filtered = user_rows[user_rows["rating"] >= min_rating]
        if filtered.empty:
            raise ValueError(
                f"User {user_id} has no ratings >= {min_rating}. Adjust the threshold or provide tag preferences."
            )

        weighted_rows: List[sparse.csr_matrix] = []
        rated_sources: List[Dict[str, object]] = []
        for _, row in filtered.iterrows():
            book_id = int(row["book_id"])
            goodreads_id = self.book_id_to_goodreads.get(book_id)
            if goodreads_id is None:
                continue

            idx = self.tag_matrices.book_id_to_idx.get(int(goodreads_id))
            if idx is None:
                continue

            rating_weight = float(row["rating"])
            weighted_rows.append(self.tfidf_matrix.getrow(idx) * rating_weight)

            book_row = self.books_lookup.loc[goodreads_id] if goodreads_id in self.books_lookup.index else None
            rated_sources.append(
                {
                    "book_id": book_id,
                    "goodreads_book_id": goodreads_id,
                    "title": None if book_row is None else (book_row.get("title") or book_row.get("original_title")),
                    "authors": None if book_row is None else book_row.get("authors", ""),
                    "rating": rating_weight,
                }
            )

        if not weighted_rows:
            raise ValueError(
                f"User {user_id} ratings could not be mapped to tag features. Check that books.csv and book_tags.csv align."
            )

        user_vec = sparse.vstack(weighted_rows).sum(axis=0)
        user_vec = sparse.csr_matrix(user_vec)
        normalized = normalize(user_vec, norm="l2", axis=1, copy=False)

        rated_sources.sort(key=lambda entry: entry.get("rating", 0.0), reverse=True)

        return UserRatingProfile(vector=normalized, rated_books=rated_sources)

    def recommend(
        self,
        *,
        preferred_tags: Optional[Iterable[int]] = None,
        preferred_tag_weights: Optional[Dict[int, float]] = None,
        top_k: int = 10,
        include_contributors: bool = True,
        user_vec: Optional[sparse.csr_matrix] = None,
    ) -> pd.DataFrame:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if user_vec is None:
            user_vec = self._make_user_vector(preferred_tags, preferred_tag_weights)

        scores = user_vec @ self.tfidf_matrix.T
        scores = np.asarray(scores.todense()).ravel()

        if not np.any(scores > 0):
            return pd.DataFrame(columns=["rank", "goodreads_book_id", "title", "authors", "score", "top_tag_contributors"])

        top_indices = np.argpartition(-scores, kth=min(top_k, scores.size - 1))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        records = []
        user_dense = user_vec.toarray().ravel()
        user_nonzero_cols = user_vec.indices

        for rank, idx in enumerate(top_indices, start=1):
            book_id = self.tag_matrices.idx_to_book_id[idx]
            book_row = self.books_lookup.loc[book_id] if book_id in self.books_lookup.index else None
            title = None if book_row is None else (book_row.get("title") or book_row.get("original_title"))
            authors = None if book_row is None else book_row.get("authors", "")

            contributors: List[Tuple[str, float]] = []
            if include_contributors and user_nonzero_cols.size > 0:
                row = self.tfidf_matrix.getrow(idx)
                intersect_cols = np.intersect1d(row.indices, user_nonzero_cols, assume_unique=False)
                if intersect_cols.size > 0:
                    vals = row[:, intersect_cols].toarray().ravel() * user_dense[intersect_cols]
                    order = np.argsort(-vals)
                    for pos in order[:5]:
                        tag_col = int(intersect_cols[pos])
                        tag_id = self.idx_to_tag_id.get(tag_col, tag_col)
                        tag_label = self.tag_lookup.get(tag_id, str(tag_id)) if self.tag_lookup is not None else str(tag_id)
                        contributors.append((tag_label, float(vals[pos])))

            records.append(
                {
                    "rank": rank,
                    "goodreads_book_id": int(book_id),
                    "title": title,
                    "authors": authors,
                    "score": float(scores[idx]),
                    "top_tag_contributors": contributors,
                }
            )

        return pd.DataFrame.from_records(records)

    def score_user_history(
        self,
        *,
        user_vec: sparse.csr_matrix,
        read_titles: Iterable[str],
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Return the read books that align best with the provided preferences."""

        matches: Dict[int, Dict[str, object]] = {}
        for raw_title in read_titles:
            title = raw_title.strip()
            if not title:
                continue
            mask = self.books_lookup["title"].str.contains(title, case=False, na=False) | self.books_lookup[
                "original_title"
            ].str.contains(title, case=False, na=False)
            for book_id in self.books_lookup[mask].index.tolist():
                matches[book_id] = {
                    "title": self.books_lookup.loc[book_id].get("title")
                    or self.books_lookup.loc[book_id].get("original_title"),
                    "authors": self.books_lookup.loc[book_id].get("authors", ""),
                }

        if not matches:
            return pd.DataFrame(columns=["title", "authors", "score", "top_tag_contributors"])

        records = []
        user_dense = user_vec.toarray().ravel()
        user_nonzero_cols = user_vec.indices

        for book_id, meta in matches.items():
            idx = self.tag_matrices.book_id_to_idx.get(int(book_id))
            if idx is None:
                continue
            score = float((user_vec @ self.tfidf_matrix[idx].T).toarray().ravel()[0])

            contributors: List[Tuple[str, float]] = []
            if user_nonzero_cols.size > 0:
                row = self.tfidf_matrix.getrow(idx)
                intersect_cols = np.intersect1d(row.indices, user_nonzero_cols, assume_unique=False)
                if intersect_cols.size > 0:
                    vals = row[:, intersect_cols].toarray().ravel() * user_dense[intersect_cols]
                    order = np.argsort(-vals)
                    for pos in order[:5]:
                        tag_col = int(intersect_cols[pos])
                        tag_id = self.idx_to_tag_id.get(tag_col, tag_col)
                        tag_label = self.tag_lookup.get(tag_id, str(tag_id)) if self.tag_lookup is not None else str(tag_id)
                        contributors.append((tag_label, float(vals[pos])))

            records.append(
                {
                    "title": meta["title"],
                    "authors": meta["authors"],
                    "score": score,
                    "top_tag_contributors": contributors,
                }
            )

        if not records:
            return pd.DataFrame(columns=["title", "authors", "score", "top_tag_contributors"])

        ordered = sorted(records, key=lambda r: r["score"], reverse=True)[:top_k]
        return pd.DataFrame.from_records(ordered)


def _parse_tag_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    cleaned = [part.strip() for part in raw.split(",") if part.strip()]
    if not cleaned:
        return None
    return [int(part) for part in cleaned]


def _parse_tag_weights(raw: Optional[str]) -> Optional[Dict[int, float]]:
    if raw is None:
        return None
    weights: Dict[int, float] = {}
    for token in raw.split(","):
        if not token.strip():
            continue
        if ":" not in token:
            raise ValueError("Tag weights must be in tag_id:weight format, separated by commas.")
        tag_part, weight_part = token.split(":", 1)
        weights[int(tag_part.strip())] = float(weight_part.strip())
    return weights or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Content-based recommender for Goodbooks-10k")
    parser.add_argument("--books-path", help="Path to books.csv (defaults to data/books.csv if present)")
    parser.add_argument("--tags-path", help="Path to tags.csv (defaults to data/tags.csv if present)")
    parser.add_argument(
        "--book-tags-path", help="Path to book_tags.csv (defaults to data/book_tags.csv if present)"
    )
    parser.add_argument("--ratings-path", help="Path to ratings.csv (defaults to data/ratings.csv if present)")
    parser.add_argument(
        "--eval-last-holdout",
        action="store_true",
        help="Run temporal last-item evaluation with ratings.csv and exit",
    )
    parser.add_argument("--eval-k", type=int, default=10, help="Cutoff k for evaluation metrics (default: 10)")
    parser.add_argument("--user-id", type=int, help="User ID from ratings.csv for history-based recommendations")
    parser.add_argument("--title", help="Book title (full or partial) to base recommendations on")
    parser.add_argument("--preferred-tags", help="Comma-separated tag_ids representing user interests")
    parser.add_argument(
        "--preferred-tag-weights",
        help="Comma-separated tag_id:weight pairs to assign different weights to preferences",
    )
    parser.add_argument("--min-count", type=int, default=1, help="Minimum tag count to keep for the matrix")
    parser.add_argument("--top-k", type=int, default=1, help="Number of recommendations to return (default: 1)")
    parser.add_argument("--text-weight", type=float, default=1.0, help="Weight applied to text features (title flow)")
    parser.add_argument("--tag-weight", type=float, default=1.0, help="Weight applied to tag features (title flow)")
    return parser.parse_args()


def _project_root() -> Path:
    """Best-effort discovery of the project root across relocated repo layouts.

    The recommender has been moved between repos (for example, nested under
    ``src/CB``) where simply walking one level up from ``__file__`` no longer
    lands on the actual repo root that contains ``data/``. We therefore scan
    ancestors for common markers (``.git`` or a ``data`` directory) and fall
    back to the parent of ``src`` when no marker is found.
    """

    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / ".git").exists() or (parent / "data").exists():
            return parent

    # Fallback: assume the repo root is one level above the package root
    return path.parents[1]


def _expand_env_path(env_var: str) -> Optional[Path]:
    """Return the path from an environment variable if it points to an existing directory."""

    raw = os.getenv(env_var)
    if not raw:
        return None

    candidate = Path(raw).expanduser().resolve()
    return candidate if candidate.exists() else None


def _resolve_data_path(
    supplied: Optional[str],
    defaults: Iterable[str],
    description: str,
    flag_hint: str,
    env_var: str = "GOODBOOKS_DATA_DIR",
) -> Path:
    """Prefer explicit CLI path; otherwise fall back to common repo and env var locations."""

    if supplied:
        path = Path(supplied).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(
            f"{description} not found at {path}. Check the path or pass a valid one via {flag_hint}."
        )

    env_base = _expand_env_path(env_var)
    search_roots = [p for p in [env_base, _project_root(), Path.cwd()] if p is not None]

    for root in search_roots:
        for rel in defaults:
            path = (root / rel).expanduser()
            if path.exists():
                return path

    raise FileNotFoundError(
        f"{description} missing. Place it at one of {list(defaults)} under {env_var}, the repo root, or the current "
        f"working directory, or provide the path explicitly with {flag_hint}."
    )


def _resolve_tag_inputs(raw: str, tags: pd.DataFrame) -> Tuple[List[int], List[str]]:
    """Resolve a comma-separated list of tag names or IDs to tag_ids.

    Returns (found_tag_ids, missing_tokens)
    """

    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return [], []

    resolved: List[int] = []
    missing: List[str] = []

    for token in tokens:
        if token.isdigit():
            tag_id = int(token)
            if (tags["tag_id"] == tag_id).any():
                resolved.append(tag_id)
            else:
                missing.append(token)
            continue

        # Name-based search (substring match)
        matches = tags[tags["tag_name"].str.contains(token, case=False, na=False)]
        if not matches.empty:
            resolved.append(int(matches.iloc[0]["tag_id"]))
        else:
            missing.append(token)

    return resolved, missing


def _interactive_tag_prompt(recommender: TagPreferenceRecommender, tags: pd.DataFrame, top_k: int) -> None:
    """Ask the user for preferred tags and display a single recommendation plus history context."""

    print("\nNo tag preferences supplied via flags. Enter them interactively below.")
    raw_tags = input("Enter tag names or IDs you like (comma-separated): ").strip()
    if not raw_tags:
        print("No input received; exiting.")
        return

    preferred_tags, missing_tokens = _resolve_tag_inputs(raw_tags, tags)
    if missing_tokens:
        print(f"Skipped unknown tags: {', '.join(missing_tokens)}")
    if not preferred_tags:
        print("No valid tags found; exiting.")
        return

    user_vec = recommender.make_user_vector(preferred_tags=preferred_tags)
    results = recommender.recommend(user_vec=user_vec, top_k=max(1, top_k), include_contributors=True)

    if results.empty:
        print("No recommendations found for the provided tag preferences.")
    else:
        _print_recommendations(results)

    raw_history = input("\nOptional: titles you've read (comma-separated, press Enter to skip): ").strip()
    if not raw_history:
        return

    history_titles = [token.strip() for token in raw_history.split(",") if token.strip()]
    if not history_titles:
        return

    history = recommender.score_user_history(user_vec=user_vec, read_titles=history_titles, top_k=5)
    if history.empty:
        print("\nNo previously read books matched those tags.")
        return

    print("\nBooks you've read that share these tags:\n")
    for _, row in history.iterrows():
        title = row["title"] or "<unknown title>"
        authors = row["authors"] or "<unknown author>"
        print(f"- {title} — {authors} (alignment score={row['score']:.3f})")
        if row["top_tag_contributors"]:
            formatted = ", ".join(f"{name}:{contrib:.3f}" for name, contrib in row["top_tag_contributors"])
            print(f"    tag overlap: {formatted}")


def _print_recommendations(results: pd.DataFrame) -> None:
    """Pretty-print the top recommendation safely and explain why it was chosen."""

    pd.set_option("display.max_colwidth", None)
    if results is None or len(results) == 0 or results.empty:
        print("No recommendations found for the provided tag preferences.")
        return

    top_row = results.iloc[0]
    title = top_row.get("title") or "<unknown title>"
    authors = top_row.get("authors") or "<unknown author>"

    rank_val = top_row.get("rank", 1)
    try:
        rank = int(rank_val if pd.notna(rank_val) else 1)
    except Exception:
        rank = 1

    score_val = top_row.get("score", 0.0)
    try:
        score = float(score_val if pd.notna(score_val) else 0.0)
    except Exception:
        score = 0.0

    print("\nTop recommendation based on tag preferences:\n")
    print(f"{rank:2d}. {title} — {authors} (score={score:.3f})")

    contribs = top_row.get("top_tag_contributors") or []
    if isinstance(contribs, (list, tuple)) and contribs:
        try:
            formatted = ", ".join(f"{name}:{float(contrib):.3f}" for name, contrib in contribs[:3])
            print(f"\nWhy this book? Strong overlap with your tag preferences: {formatted}")
        except Exception:
            print("\nWhy this book? It best matches the tag signals you provided.")
    else:
        print("\nWhy this book? It best matches the tag signals you provided.")


def _print_rating_based_recommendation(results: pd.DataFrame, rated_books: List[Dict[str, object]]) -> None:
    """Print a single recommendation derived from ratings and the books that informed it."""

    pd.set_option("display.max_colwidth", None)
    if results is None or results.empty:
        print("No recommendation could be generated from the provided ratings.")
        return

    top_row = results.iloc[0]
    title = top_row.get("title") or "<unknown title>"
    authors = top_row.get("authors") or "<unknown author>"
    score_val = top_row.get("score", 0.0)
    try:
        score = float(score_val if pd.notna(score_val) else 0.0)
    except Exception:
        score = 0.0

    print("\nTop recommendation based on your past ratings:\n")
    print(f"1. {title} — {authors} (score={score:.3f})")

    if rated_books:
        print("\nBooks that influenced this pick:")
        for entry in rated_books:
            src_title = entry.get("title") or "<unknown title>"
            src_authors = entry.get("authors") or "<unknown author>"
            src_rating = entry.get("rating")
            book_id = entry.get("book_id")
            goodreads_id = entry.get("goodreads_book_id")
            rating_part = f", rating={src_rating}" if src_rating is not None else ""
            print(
                f"- {src_title} — {src_authors} (book_id={book_id}, goodreads_id={goodreads_id}{rating_part})"
            )

    contributors = top_row.get("top_tag_contributors") or []
    if contributors:
        tag_summary = ", ".join(f"{name}:{float(contrib):.3f}" for name, contrib in contributors[:3])
        print(f"\nWhy this book? Strong tag overlap with your highly rated reads: {tag_summary}")
    else:
        print("\nWhy this book? It best matches the tag signals inferred from your highly rated books.")


def _run_preference_flow(recommender: TagPreferenceRecommender, tags: pd.DataFrame, args: argparse.Namespace) -> None:
    """Drive the tag-preference path with optional interactivity."""

    if args.preferred_tags is None and args.preferred_tag_weights is None:
        _interactive_tag_prompt(recommender, tags, args.top_k)
        return

    results = recommender.recommend(
        preferred_tags=args.preferred_tags,
        preferred_tag_weights=args.preferred_tag_weights,
        top_k=args.top_k,
        include_contributors=True,
    )
    _print_recommendations(results)


def _print_eval_results(eval_real, eval_zero, k: int) -> None:
    print("\n=== Temporal Last-Item Evaluation ===")
    for line in format_eval_summary(eval_real, k):
        print(line)

    print("\n=== Sanity: Remove User History (should drop) ===")
    for line in format_eval_summary(eval_zero, k):
        print(line)

    if np.isfinite(eval_real.hit_at_k) and np.isfinite(eval_zero.hit_at_k):
        if eval_real.hit_at_k > max(0.01, eval_zero.hit_at_k + 0.02):
            print("\nPASS: Model uses past ratings to improve next-book recommendation.")
        else:
            print("\nWARN: Little/no gain over no-history control. Check leakage or model logic.")


def _print_eval_with_explanation(eval_real, eval_zero, k: int) -> None:
    """Print metrics plus a short explanation of what the numbers mean."""

    _print_eval_results(eval_real, eval_zero, k)

    if not (
        np.isfinite(eval_real.hit_at_k)
        and np.isfinite(eval_zero.hit_at_k)
        and np.isfinite(eval_real.mrr_at_k)
        and np.isfinite(eval_zero.mrr_at_k)
    ):
        print("\nEvaluation explanation unavailable because metrics are undefined (likely due to sparse data).")
        return

    delta_hit = eval_real.hit_at_k - eval_zero.hit_at_k
    delta_mrr = eval_real.mrr_at_k - eval_zero.mrr_at_k

    print(
        "\nExplanation: Hit@{k} measures how often the held-out last book appears in the top {k}; MRR@{k}"
        " rewards higher ranks. Compared to the no-history control, Hit@{k} changed by {dh:+.4f} and"
        " MRR@{k} changed by {dm:+.4f}, showing how much the model benefits from using each user's past ratings."
        .format(k=k, dh=delta_hit, dm=delta_mrr)
    )


def _run_evaluation_and_explain(ratings: pd.DataFrame, k: int) -> None:
    """Run temporal evaluation and print an explanation for the observed metrics."""

    prepared = prepare_ratings(ratings)
    eval_real, eval_zero = run_temporal_evaluation(prepared, k=k)
    _print_eval_with_explanation(eval_real, eval_zero, k)


def run_cli():
    args = _parse_args()

    ratings_needed = args.eval_last_holdout or args.user_id is not None or (
        args.preferred_tags is None and args.preferred_tag_weights is None and not args.title
    )
    ratings_path = None
    ratings = None
    if ratings_needed:
        ratings_path = _resolve_data_path(
            args.ratings_path,
            [
                "data/goodbooks-10k/ratings.csv",
                "data/ratings.csv",
                "goodbooks-10k/ratings.csv",
                "ratings.csv",
            ],
            "ratings.csv",
            "--ratings-path",
        )
        ratings = load_ratings(str(ratings_path))

    if args.eval_last_holdout:
        if ratings is None:
            raise FileNotFoundError("ratings.csv is required when --eval-last-holdout is set.")
        _run_evaluation_and_explain(ratings, k=max(1, args.eval_k))
        return

    books_path = _resolve_data_path(
        args.books_path,
        ["data/goodbooks-10k/books.csv", "data/books.csv", "goodbooks-10k/books.csv", "books.csv"],
        "books.csv",
        "--books-path",
    )
    tags_path = _resolve_data_path(
        args.tags_path,
        ["data/goodbooks-10k/tags.csv", "data/tags.csv", "goodbooks-10k/tags.csv", "tags.csv"],
        "tags.csv",
        "--tags-path",
    )
    book_tags_path = _resolve_data_path(
        args.book_tags_path,
        ["data/goodbooks-10k/book_tags.csv", "data/book_tags.csv", "goodbooks-10k/book_tags.csv", "book_tags.csv"],
        "book_tags.csv",
        "--book-tags-path",
    )

    books = load_books(str(books_path))
    tags = load_tags(str(tags_path))
    book_tags = load_book_tags(str(book_tags_path))
    ratings = load_ratings(str(ratings_path)) if ratings_path is not None else None

    tag_matrix = build_tag_matrix(tags, book_tags, min_count=args.min_count)

    # Branch: title-based similarity or tag-preference recommendations
    if args.title:
        features, ordered_books = build_feature_matrix(
            books,
            tag_matrix,
            text_weight=args.text_weight,
            tag_weight=args.tag_weight,
        )
        recommender = ContentBasedRecommender(features, ordered_books)
        results = recommender.recommend_from_title(args.title, top_k=max(1, args.top_k))

        if not results:
            print(f"No books match the query '{args.title}'.")
            return

        top_title, top_authors, top_score = results[0]
        print("\nTop recommendation based on your seed title:\n")
        print(f"1. {top_title} — {top_authors} (score={top_score:.3f})")
        print("\nWhy this book? It has the strongest combined text/tag similarity to your provided title.")
        return

    preferred_tags = _parse_tag_list(args.preferred_tags)
    preferred_tag_weights = _parse_tag_weights(args.preferred_tag_weights)
    args.preferred_tags = preferred_tags
    args.preferred_tag_weights = preferred_tag_weights
    recommender = TagPreferenceRecommender(tag_matrix, books, tags)

    if args.user_id is not None:
        if ratings is None:
            raise FileNotFoundError("ratings.csv is required when --user-id is provided.")
        profile = recommender.build_user_profile_from_ratings(ratings, args.user_id)
        results = recommender.recommend(user_vec=profile.vector, top_k=1, include_contributors=True)
        _print_rating_based_recommendation(results, profile.rated_books)
        _run_evaluation_and_explain(ratings, k=max(1, args.eval_k))
        return

    if ratings is not None and preferred_tags is None and preferred_tag_weights is None:
        raw_id = input("Enter a user_id from ratings.csv (press Enter to skip): ").strip()
        if raw_id:
            try:
                user_id = int(raw_id)
                profile = recommender.build_user_profile_from_ratings(ratings, user_id)
                results = recommender.recommend(user_vec=profile.vector, top_k=1, include_contributors=True)
                _print_rating_based_recommendation(results, profile.rated_books)
                _run_evaluation_and_explain(ratings, k=max(1, args.eval_k))
                return
            except Exception as exc:  # fallback to tag prompt
                print(f"Could not build recommendations from user_id {raw_id}: {exc}")

    _run_preference_flow(recommender, tags, args)


if __name__ == "__main__":
    run_cli()
