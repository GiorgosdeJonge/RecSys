"""Data loading utilities for the Goodbooks-10k dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class TagMatrices:
    """Container for the sparse tag matrix and lookup dictionaries."""

    matrix: sparse.csr_matrix
    book_id_to_idx: Dict[int, int]
    idx_to_book_id: Dict[int, int]
    tag_id_to_idx: Dict[int, int]


def load_books(path: str) -> pd.DataFrame:
    """Load the books table.

    The Goodbooks-10k ``books.csv`` file includes columns such as ``book_id``,
    ``goodreads_book_id``, ``original_title``, and ``authors``. This helper
    keeps only a subset that is helpful for content-based recommendations.
    """

    books = pd.read_csv(path)
    # Keep core fields and fill missing titles/authors with empty strings so TF–IDF works.
    columns = [
        "book_id",
        "goodreads_book_id",
        "original_title",
        "authors",
        "title",
    ]
    available_cols = [col for col in columns if col in books.columns]
    books = books[available_cols].copy()
    for text_col in ["original_title", "authors", "title"]:
        if text_col in books:
            books[text_col] = books[text_col].fillna("")
    return books


def load_tags(path: str) -> pd.DataFrame:
    """Load the tag vocabulary."""

    tags = pd.read_csv(path)
    expected = {"tag_id", "tag_name"}
    missing = expected.difference(tags.columns)
    if missing:
        raise ValueError(f"tags.csv is missing columns: {missing}")
    return tags


def load_book_tags(path: str) -> pd.DataFrame:
    """Load the book-to-tag assignments."""

    book_tags = pd.read_csv(path)
    expected = {"goodreads_book_id", "tag_id", "count"}
    missing = expected.difference(book_tags.columns)
    if missing:
        raise ValueError(f"book_tags.csv is missing columns: {missing}")
    return book_tags


def build_tag_matrix(
    tags: pd.DataFrame,
    book_tags: pd.DataFrame,
    *,
    min_count: int = 1,
) -> TagMatrices:
    """Create a sparse book-tag matrix.

    Args:
        tags: Tag vocabulary with ``tag_id`` and ``tag_name`` columns.
        book_tags: Mapping of ``goodreads_book_id`` to tag counts.
        min_count: Minimum count threshold for a tag assignment to be kept.

    Returns:
        TagMatrices: sparse CSR matrix with shape (num_books, num_tags).
    """

    filtered = book_tags[book_tags["count"] >= min_count].copy()
    # Create index mappings for stable matrix construction.
    unique_book_ids = np.sort(filtered["goodreads_book_id"].unique())
    unique_tag_ids = np.sort(filtered["tag_id"].unique())

    book_id_to_idx = {book_id: idx for idx, book_id in enumerate(unique_book_ids)}
    tag_id_to_idx = {tag_id: idx for idx, tag_id in enumerate(unique_tag_ids)}

    row_indices = filtered["goodreads_book_id"].map(book_id_to_idx)
    col_indices = filtered["tag_id"].map(tag_id_to_idx)
    data = filtered["count"].astype(float)

    matrix = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(unique_book_ids), len(unique_tag_ids)),
    ).tocsr()

    return TagMatrices(
        matrix=matrix,
        book_id_to_idx=book_id_to_idx,
        idx_to_book_id={idx: book_id for book_id, idx in book_id_to_idx.items()},
        tag_id_to_idx=tag_id_to_idx,
    )
