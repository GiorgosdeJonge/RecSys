"""CLI for a content-based Goodbooks-10k recommender."""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .data_loader import TagMatrices, build_tag_matrix, load_book_tags, load_books, load_tags
from .feature_builder import build_feature_matrix


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
    parser.add_argument("--books-path", required=True, help="Path to books.csv")
    parser.add_argument("--tags-path", required=True, help="Path to tags.csv")
    parser.add_argument("--book-tags-path", required=True, help="Path to book_tags.csv")
    parser.add_argument("--title", help="Book title (full or partial) to base recommendations on")
    parser.add_argument("--preferred-tags", help="Comma-separated tag_ids representing user interests")
    parser.add_argument(
        "--preferred-tag-weights",
        help="Comma-separated tag_id:weight pairs to assign different weights to preferences",
    )
    parser.add_argument("--min-count", type=int, default=1, help="Minimum tag count to keep for the matrix")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return")
    parser.add_argument("--text-weight", type=float, default=1.0, help="Weight applied to text features (title flow)")
    parser.add_argument("--tag-weight", type=float, default=1.0, help="Weight applied to tag features (title flow)")
    return parser.parse_args()


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
    """Ask the user for preferred tags and display recommendations plus history context."""

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
    results = recommender.recommend(user_vec=user_vec, top_k=top_k, include_contributors=True)

    if results.empty:
        print("No recommendations found for the provided tag preferences.")
    else:
        print("\nTop recommendations based on your tags:\n")
        for _, row in results.iterrows():
            title = row["title"] or "<unknown title>"
            authors = row["authors"] or "<unknown author>"
            print(f"{row['rank']:2d}. {title} — {authors} (score={row['score']:.3f})")
            if row["top_tag_contributors"]:
                formatted = ", ".join(f"{name}:{contrib:.3f}" for name, contrib in row["top_tag_contributors"])
                print(f"    top tag contributors: {formatted}")

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


def main():
    args = _parse_args()

    books = load_books(args.books_path)
    tags = load_tags(args.tags_path)
    book_tags = load_book_tags(args.book_tags_path)

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
        results = recommender.recommend_from_title(args.title, top_k=args.top_k)

        for i, (title, authors, score) in enumerate(results, start=1):
            print(f"{i:2d}. {title} — {authors} (score={score:.3f})")
        return

    preferred_tags = _parse_tag_list(args.preferred_tags)
    preferred_tag_weights = _parse_tag_weights(args.preferred_tag_weights)
    recommender = TagPreferenceRecommender(tag_matrix, books, tags)

    # Interactive branch when no preferences were provided via flags
    if preferred_tags is None and preferred_tag_weights is None:
        _interactive_tag_prompt(recommender, tags, args.top_k)
        return

    results = recommender.recommend(
        preferred_tags=preferred_tags,
        preferred_tag_weights=preferred_tag_weights,
        top_k=args.top_k,
        include_contributors=True,
    )

    pd.set_option("display.max_colwidth", None)
    if results.empty:
        print("No recommendations found for the provided tag preferences.")
        return

    print("\nTop recommendations based on tag preferences:\n")
    for _, row in results.iterrows():
        title = row["title"] or "<unknown title>"
        authors = row["authors"] or "<unknown author>"
        print(f"{row['rank']:2d}. {title} — {authors} (score={row['score']:.3f})")
        if row["top_tag_contributors"]:
            formatted = ", ".join(f"{name}:{contrib:.3f}" for name, contrib in row["top_tag_contributors"])
            print(f"    top tag contributors: {formatted}")


if __name__ == "__main__":
    main()
