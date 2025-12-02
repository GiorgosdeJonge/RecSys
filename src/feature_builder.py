"""Feature builders for content-based recommendations."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from .data_loader import TagMatrices


def _combine_text_columns(books: pd.DataFrame) -> List[str]:
    title = books.get("original_title", books.get("title", ""))
    authors = books.get("authors", "")
    return (title.astype(str) + " " + authors.astype(str)).tolist()


def build_text_features(
    books: pd.DataFrame,
    *,
    max_features: int = 6000,
    ngram_range: tuple = (1, 2),
) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    """Create TF–IDF vectors from titles and authors."""

    corpus = _combine_text_columns(books)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        lowercase=True,
    )
    features = vectorizer.fit_transform(corpus)
    return features, vectorizer


def build_feature_matrix(
    books: pd.DataFrame,
    tag_matrices: TagMatrices,
    *,
    text_weight: float = 1.0,
    tag_weight: float = 1.0,
    max_text_features: int = 6000,
) -> Tuple[sparse.csr_matrix, pd.DataFrame]:
    """Combine text and tag features into a single sparse matrix.

    Returns both the matrix and the ordered books dataframe so downstream
    components can reference titles and authors by row index.
    """

    # Align books to the order used in the tag matrix
    book_index = pd.Index([tag_matrices.idx_to_book_id[i] for i in range(tag_matrices.matrix.shape[0])])
    books_lookup = books.set_index("goodreads_book_id")
    aligned_books = books_lookup.loc[book_index].reset_index()

    text_features, _ = build_text_features(
        aligned_books,
        max_features=max_text_features,
    )

    transformer = TfidfTransformer()
    tag_features = transformer.fit_transform(tag_matrices.matrix)

    if text_weight != 1.0:
        text_features = text_features * text_weight
    if tag_weight != 1.0:
        tag_features = tag_features * tag_weight

    combined = sparse.hstack([text_features, tag_features]).tocsr()
    return combined, aligned_books
