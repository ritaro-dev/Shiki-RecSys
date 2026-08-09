from unittest.mock import MagicMock

import pandas as pd

from shiki_recsys.retrievers.common import RetrieverName
from shiki_recsys.training.ranker import build_ranker_training_data


def _candidates(source: RetrieverName) -> pd.DataFrame:
    """Создаёт тестовую выдачу retriever-а."""

    return pd.DataFrame(
        {
            "anime_id": pd.Series([10, 20], dtype="int64"),
            "score": pd.Series([2.0, 1.0], dtype="float64"),
            "source": pd.Series(
                [source.value, source.value],
                dtype="string",
            ),
            "source_rank": pd.Series([1, 2], dtype="int32"),
        }
    )


def test_build_ranker_training_data_combines_users() -> None:
    """Проверяет сбор размеченных данных нескольких пользователей."""

    popularity = MagicMock()
    explicit_svd = MagicMock()
    implicit_als = MagicMock()
    content_tfidf = MagicMock()

    popularity.retrieve.return_value = _candidates(RetrieverName.POPULARITY)
    explicit_svd.retrieve.return_value = _candidates(RetrieverName.EXPLICIT_SVD)
    implicit_als.retrieve.return_value = _candidates(RetrieverName.IMPLICIT_ALS)
    content_tfidf.retrieve.return_value = _candidates(RetrieverName.CONTENT_TFIDF)

    data = build_ranker_training_data(
        user_ids=[1, 2],
        known_items_by_user={
            1: {100},
            2: {200},
        },
        positive_items_by_user={
            1: {10},
            2: {20},
        },
        candidate_count=2,
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
    )

    assert data["user_id"].tolist() == [1, 1, 2, 2]
    assert data["anime_id"].tolist() == [10, 20, 10, 20]
    assert data["target"].tolist() == [1, 0, 0, 1]

    explicit_svd.retrieve.assert_any_call(
        user_id=1,
        candidate_count=2,
        exclude_anime_ids={100},
    )
    explicit_svd.retrieve.assert_any_call(
        user_id=2,
        candidate_count=2,
        exclude_anime_ids={200},
    )
