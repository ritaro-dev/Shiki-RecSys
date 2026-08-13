from unittest.mock import Mock

import pandas as pd
import pytest

from shiki_recsys.config.inference import RecommendationServingConfig
from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.inference.recommendation_service import build_recommendations
from shiki_recsys.inference.user_state import UserState
from shiki_recsys.model_artifacts import ArtifactInferenceConfig


@pytest.fixture
def artifact_config() -> ArtifactInferenceConfig:
    return ArtifactInferenceConfig(
        retrieval_k=100,
        positive_rating_threshold=8,
        max_positive_items=50,
    )


@pytest.fixture
def serving_config() -> RecommendationServingConfig:
    return RecommendationServingConfig(
        top_k=2,
        min_positive_items=5,
    )


def _bundle() -> ModelBundle:
    """Создаёт model bundle для тестов inference routing."""
    return ModelBundle(
        popularity=Mock(),
        explicit_svd=Mock(),
        implicit_als=Mock(),
        content_tfidf=Mock(),
        ranker=Mock(),
    )


@pytest.mark.parametrize(
    ("user_exists", "history_synced", "expected_state"),
    [
        (False, False, UserState.USER_NOT_FOUND),
        (True, False, UserState.NOT_SYNCED),
    ],
)
def test_build_recommendations_returns_empty_for_unavailable_user_state(
    user_exists: bool,
    history_synced: bool,
    expected_state: UserState,
    artifact_config: ArtifactInferenceConfig,
    serving_config: RecommendationServingConfig,
) -> None:
    """Проверяет состояния без доступного recommendation inference."""
    result = build_recommendations(
        user_id=1,
        interactions=pd.DataFrame(),
        user_exists=user_exists,
        history_synced=history_synced,
        bundle=_bundle(),
        artifact_config=artifact_config,
        serving_config=serving_config,
    )

    assert result.state == expected_state
    assert result.recommendations.empty
    assert result.recommendations.columns.tolist() == [
        "anime_id",
        "rank",
    ]


def test_build_recommendations_routes_cold_user(
    monkeypatch: pytest.MonkeyPatch,
    artifact_config: ArtifactInferenceConfig,
    serving_config: RecommendationServingConfig,
) -> None:
    """Проверяет cold routing и final top-K."""
    bundle = _bundle()
    monkeypatch.setattr(
        ModelBundle,
        "supports_personal_user",
        Mock(return_value=False),
    )

    monkeypatch.setattr(
        "shiki_recsys.inference.recommendation_service.count_supported_positive_items",
        Mock(return_value=2),
    )

    cold_ranking = pd.DataFrame(
        {
            "anime_id": [30, 40, 50],
            "cold_score": [1.5, 1.0, 0.5],
            "rank": [1, 2, 3],
        }
    )
    cold_mock = Mock(return_value=cold_ranking)

    monkeypatch.setattr(
        "shiki_recsys.inference.recommendation_service.build_cold_ranking",
        cold_mock,
    )

    interactions = pd.DataFrame(
        {
            "anime_id": [10, 20],
            "rating": [9.0, 8.0],
        }
    )

    result = build_recommendations(
        user_id=1,
        interactions=interactions,
        user_exists=True,
        history_synced=True,
        bundle=bundle,
        artifact_config=artifact_config,
        serving_config=serving_config,
    )

    assert result.state == UserState.SPARSE_COLD
    assert result.recommendations["anime_id"].tolist() == [30, 40]

    cold_mock.assert_called_once_with(
        state=UserState.SPARSE_COLD,
        interactions=interactions,
        known_anime_ids={10, 20},
        candidate_count=100,
        relevance_threshold=8,
        max_positive_items=50,
        popularity=bundle.popularity,
        content_tfidf=bundle.content_tfidf,
    )


def test_build_recommendations_routes_warm_user(
    monkeypatch: pytest.MonkeyPatch,
    artifact_config: ArtifactInferenceConfig,
    serving_config: RecommendationServingConfig,
) -> None:
    """Проверяет routing пользователя в warm pipeline."""
    bundle = _bundle()
    monkeypatch.setattr(
        ModelBundle,
        "supports_personal_user",
        Mock(return_value=True),
    )

    monkeypatch.setattr(
        "shiki_recsys.inference.recommendation_service.count_supported_positive_items",
        Mock(return_value=0),
    )

    warm_ranking = pd.DataFrame(
        {
            "anime_id": [30, 40],
            "ranker_score": [2.0, 1.0],
            "rank": [1, 2],
        }
    )
    warm_mock = Mock(return_value=warm_ranking)

    monkeypatch.setattr(
        "shiki_recsys.inference.recommendation_service.build_warm_ranking",
        warm_mock,
    )

    interactions = pd.DataFrame(
        {
            "anime_id": [10],
            "rating": [6.0],
        }
    )

    result = build_recommendations(
        user_id=1,
        interactions=interactions,
        user_exists=True,
        history_synced=True,
        bundle=bundle,
        artifact_config=artifact_config,
        serving_config=serving_config,
    )

    assert result.state == UserState.WARM
    assert result.recommendations["anime_id"].tolist() == [30, 40]

    warm_mock.assert_called_once_with(
        user_id=1,
        known_anime_ids={10},
        candidate_count=100,
        popularity=bundle.popularity,
        explicit_svd=bundle.explicit_svd,
        implicit_als=bundle.implicit_als,
        content_tfidf=bundle.content_tfidf,
        ranker=bundle.ranker,
    )
