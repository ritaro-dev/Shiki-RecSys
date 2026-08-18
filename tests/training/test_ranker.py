from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import shiki_recsys.training.ranker as ranker_module
from shiki_recsys.retrievers.common import RetrieverName
from shiki_recsys.training.ranker import (
    build_ranker_training_data,
    fit_ranker,
)


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


def test_fit_ranker_uses_validation_positives(monkeypatch) -> None:
    """Verify ranker training from validation targets."""
    history_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "anime_id": [10, 20, 30],
        }
    )
    target_interactions = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "anime_id": [40, 50, 60],
            "rating": [8, 5, 9],
        }
    )

    config = SimpleNamespace(
        random_seed=42,
        target=SimpleNamespace(
            positive_rating_threshold=8.0,
        ),
        candidate_generation=SimpleNamespace(
            retrieval_k=100,
        ),
        ranker=object(),
    )

    training_data = pd.DataFrame(
        {
            "user_id": [1, 3],
            "anime_id": [40, 60],
            "target": [1, 1],
        }
    )

    build_training_data = MagicMock(
        return_value=training_data,
    )
    ranker = MagicMock()
    ranker_factory = MagicMock(return_value=ranker)

    monkeypatch.setattr(
        ranker_module,
        "build_ranker_training_data",
        build_training_data,
    )
    monkeypatch.setattr(
        ranker_module,
        "CatBoostRankerModel",
        ranker_factory,
    )

    popularity = MagicMock()
    explicit_svd = MagicMock()
    implicit_als = MagicMock()
    content_tfidf = MagicMock()

    result = fit_ranker(
        history_interactions=history_interactions,
        target_interactions=target_interactions,
        config=config,
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
    )

    assert result is ranker

    build_training_data.assert_called_once_with(
        user_ids={1: {40}, 3: {60}},
        known_items_by_user={
            1: {10, 20},
            2: {30},
        },
        positive_items_by_user={
            1: {40},
            3: {60},
        },
        candidate_count=100,
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
    )

    ranker_factory.assert_called_once_with(
        config.ranker,
        random_seed=42,
    )
    ranker.fit.assert_called_once_with(training_data)


def test_fit_ranker_rejects_validation_without_positives() -> None:
    """Reject ranker training when validation has no positive targets."""
    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
        }
    )
    validation = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [20],
            "rating": [5],
        }
    )

    config = SimpleNamespace(
        target=SimpleNamespace(
            positive_rating_threshold=8.0,
        )
    )

    with pytest.raises(
        ValueError,
        match="Target interactions contain no positive items.",
    ):
        fit_ranker(
            history_interactions=interactions,
            target_interactions=validation,
            config=config,
            popularity=MagicMock(),
            explicit_svd=MagicMock(),
            implicit_als=MagicMock(),
            content_tfidf=MagicMock(),
        )
