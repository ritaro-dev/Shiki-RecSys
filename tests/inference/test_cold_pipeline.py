from unittest.mock import Mock

import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from shiki_recsys.inference.cold_pipeline import (
    build_cold_ranking,
    build_new_user_ranking,
)
from shiki_recsys.inference.user_state import UserState
from shiki_recsys.retrievers.content_tfidf import ContentTFIDFRetriever
from shiki_recsys.retrievers.popularity import PopularityRetriever


def _candidates(
    anime_ids: list[int],
    source_ranks: list[int],
) -> pd.DataFrame:
    """Создаёт минимальный candidate frame для теста."""
    return pd.DataFrame(
        {
            "anime_id": anime_ids,
            "source_rank": source_ranks,
        }
    )


def test_build_new_user_ranking_fuses_content_and_popularity(
    monkeypatch,
) -> None:
    """Проверяет reciprocal-rank fusion cold-кандидатов."""
    profile = csr_matrix([[1.0, 0.0]])

    monkeypatch.setattr(
        "shiki_recsys.inference.cold_pipeline.build_content_profile",
        Mock(return_value=profile),
    )

    content_tfidf = Mock(spec=ContentTFIDFRetriever)
    content_tfidf.item_features = Mock()
    content_tfidf.retrieve_from_profile.return_value = _candidates(
        [10, 20],
        [1, 2],
    )

    popularity = Mock(spec=PopularityRetriever)
    popularity.retrieve.return_value = _candidates(
        [20, 30],
        [1, 2],
    )

    ranking = build_new_user_ranking(
        interactions=pd.DataFrame(),
        known_anime_ids={99},
        candidate_count=2,
        relevance_threshold=8,
        max_positive_items=50,
        popularity=popularity,
        content_tfidf=content_tfidf,
    )

    assert ranking["anime_id"].tolist() == [20, 10, 30]
    assert ranking["cold_score"].tolist() == [1.5, 1.0, 0.5]
    assert ranking["rank"].tolist() == [1, 2, 3]

    content_tfidf.retrieve_from_profile.assert_called_once_with(
        profile=profile,
        candidate_count=2,
        exclude_anime_ids={99},
    )
    popularity.retrieve.assert_called_once_with(
        candidate_count=2,
        exclude_anime_ids={99},
    )


def test_build_new_user_ranking_breaks_score_ties_by_anime_id(
    monkeypatch,
) -> None:
    """Проверяет детерминированный порядок при равных cold scores."""
    monkeypatch.setattr(
        "shiki_recsys.inference.cold_pipeline.build_content_profile",
        Mock(return_value=csr_matrix([[1.0]])),
    )

    content_tfidf = Mock(spec=ContentTFIDFRetriever)
    content_tfidf.item_features = Mock()
    content_tfidf.retrieve_from_profile.return_value = _candidates(
        [20],
        [1],
    )

    popularity = Mock(spec=PopularityRetriever)
    popularity.retrieve.return_value = _candidates(
        [10],
        [1],
    )

    ranking = build_new_user_ranking(
        interactions=pd.DataFrame(),
        known_anime_ids=set(),
        candidate_count=1,
        relevance_threshold=8,
        max_positive_items=50,
        popularity=popularity,
        content_tfidf=content_tfidf,
    )

    assert ranking["anime_id"].tolist() == [10, 20]
    assert ranking["cold_score"].tolist() == [1.0, 1.0]
    assert ranking["rank"].tolist() == [1, 2]


def test_build_cold_ranking_uses_popularity_without_preference_signal() -> None:
    """Проверяет popularity ranking без preference signal."""
    popularity = Mock(spec=PopularityRetriever)
    popularity.retrieve.return_value = pd.DataFrame(
        {
            "anime_id": [10, 20],
            "score": [100.0, 80.0],
            "source": ["popularity", "popularity"],
            "source_rank": [1, 2],
        }
    )

    content_tfidf = Mock(spec=ContentTFIDFRetriever)

    ranking = build_cold_ranking(
        state=UserState.NO_PREFERENCE_SIGNAL,
        interactions=pd.DataFrame(),
        known_anime_ids={30},
        candidate_count=2,
        relevance_threshold=8,
        max_positive_items=50,
        popularity=popularity,
        content_tfidf=content_tfidf,
    )

    assert ranking.columns.tolist() == [
        "anime_id",
        "cold_score",
        "rank",
    ]
    assert ranking["anime_id"].tolist() == [10, 20]
    assert ranking["cold_score"].tolist() == [100.0, 80.0]
    assert ranking["rank"].tolist() == [1, 2]

    popularity.retrieve.assert_called_once_with(
        candidate_count=2,
        exclude_anime_ids={30},
    )
    content_tfidf.retrieve_from_profile.assert_not_called()


def test_build_cold_ranking_rejects_warm_state() -> None:
    """Проверяет запрет warm-состояния в cold pipeline."""
    with pytest.raises(
        ValueError,
        match="не относится к cold inference",
    ):
        build_cold_ranking(
            state=UserState.WARM,
            interactions=pd.DataFrame(),
            known_anime_ids=set(),
            candidate_count=2,
            relevance_threshold=8,
            max_positive_items=50,
            popularity=Mock(spec=PopularityRetriever),
            content_tfidf=Mock(spec=ContentTFIDFRetriever),
        )
