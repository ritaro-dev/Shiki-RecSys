from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from shiki_recsys.retrievers.common import RetrieverName, empty_candidates
from shiki_recsys.retrievers.generation import (
    build_candidate_union,
    generate_retriever_candidates,
    score_candidate_union,
)


def test_generate_retriever_candidates_calls_all_retrievers() -> None:
    """Проверяет генерацию кандидатов всеми retriever-ами."""

    popularity = MagicMock()
    explicit_svd = MagicMock()
    implicit_als = MagicMock()
    content_tfidf = MagicMock()

    popularity_candidates = empty_candidates()
    explicit_svd_candidates = empty_candidates()
    implicit_als_candidates = empty_candidates()
    content_tfidf_candidates = empty_candidates()

    popularity.retrieve.return_value = popularity_candidates
    explicit_svd.retrieve.return_value = explicit_svd_candidates
    implicit_als.retrieve.return_value = implicit_als_candidates
    content_tfidf.retrieve.return_value = content_tfidf_candidates

    known_anime_ids = {10, 20}

    result = generate_retriever_candidates(
        user_id=7,
        known_anime_ids=known_anime_ids,
        candidate_count=100,
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
    )

    assert result == {
        RetrieverName.POPULARITY: popularity_candidates,
        RetrieverName.EXPLICIT_SVD: explicit_svd_candidates,
        RetrieverName.IMPLICIT_ALS: implicit_als_candidates,
        RetrieverName.CONTENT_TFIDF: content_tfidf_candidates,
    }

    popularity.retrieve.assert_called_once_with(
        candidate_count=100,
        exclude_anime_ids=known_anime_ids,
    )

    for retriever in (
        explicit_svd,
        implicit_als,
        content_tfidf,
    ):
        retriever.retrieve.assert_called_once_with(
            user_id=7,
            candidate_count=100,
            exclude_anime_ids=known_anime_ids,
        )


def test_build_candidate_union_deduplicates_candidates() -> None:
    """Проверяет объединение выдач retriever-ов без дубликатов."""

    popularity = pd.DataFrame(
        {
            "anime_id": [10, 20, 30],
        }
    )
    explicit_svd = pd.DataFrame(
        {
            "anime_id": [20, 40, 10],
        }
    )
    implicit_als = pd.DataFrame(
        {
            "anime_id": [50, 30],
        }
    )

    anime_ids = build_candidate_union(
        {
            RetrieverName.IMPLICIT_ALS: implicit_als,
            RetrieverName.EXPLICIT_SVD: explicit_svd,
            RetrieverName.POPULARITY: popularity,
        }
    )

    np.testing.assert_array_equal(
        anime_ids,
        np.array(
            [10, 20, 30, 40, 50],
            dtype=np.int64,
        ),
    )


def test_score_candidate_union_reuses_candidate_scores() -> None:
    """Проверяет досчёт только отсутствующих scores."""

    anime_ids = np.array(
        [10, 20, 30],
        dtype=np.int64,
    )

    retriever_candidates = {
        RetrieverName.POPULARITY: pd.DataFrame(
            {
                "anime_id": [10],
                "score": [1.0],
            }
        ),
        RetrieverName.EXPLICIT_SVD: pd.DataFrame(
            {
                "anime_id": [20],
                "score": [2.0],
            }
        ),
        RetrieverName.IMPLICIT_ALS: pd.DataFrame(
            {
                "anime_id": [30],
                "score": [3.0],
            }
        ),
        RetrieverName.CONTENT_TFIDF: pd.DataFrame(
            {
                "anime_id": [20],
                "score": [4.0],
            }
        ),
    }

    popularity = MagicMock()
    explicit_svd = MagicMock()
    implicit_als = MagicMock()
    content_tfidf = MagicMock()

    popularity.score_items.return_value = np.array([1.2, 1.3])
    explicit_svd.score_items.return_value = np.array([2.1, 2.3])
    implicit_als.score_items.return_value = np.array([3.1, 3.2])
    content_tfidf.score_items.return_value = np.array([4.1, 4.3])

    result = score_candidate_union(
        user_id=7,
        anime_ids=anime_ids,
        retriever_candidates=retriever_candidates,
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
    )

    np.testing.assert_allclose(
        result[RetrieverName.POPULARITY],
        [1.0, 1.2, 1.3],
    )
    np.testing.assert_allclose(
        result[RetrieverName.EXPLICIT_SVD],
        [2.1, 2.0, 2.3],
    )
    np.testing.assert_allclose(
        result[RetrieverName.IMPLICIT_ALS],
        [3.1, 3.2, 3.0],
    )
    np.testing.assert_allclose(
        result[RetrieverName.CONTENT_TFIDF],
        [4.1, 4.0, 4.3],
    )

    popularity.score_items.assert_called_once()
    np.testing.assert_array_equal(
        popularity.score_items.call_args.kwargs["anime_ids"],
        [20, 30],
    )

    explicit_svd.score_items.assert_called_once()
    np.testing.assert_array_equal(
        explicit_svd.score_items.call_args.kwargs["anime_ids"],
        [10, 30],
    )

    implicit_als.score_items.assert_called_once()
    np.testing.assert_array_equal(
        implicit_als.score_items.call_args.kwargs["anime_ids"],
        [10, 20],
    )

    content_tfidf.score_items.assert_called_once()
    np.testing.assert_array_equal(
        content_tfidf.score_items.call_args.kwargs["anime_ids"],
        [10, 30],
    )
