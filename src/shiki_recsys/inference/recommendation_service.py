from dataclasses import dataclass

import pandas as pd

from shiki_recsys.config.inference import RecommendationServingConfig
from shiki_recsys.features.content_users import count_supported_positive_items
from shiki_recsys.inference.cold_pipeline import build_cold_ranking
from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.inference.user_state import UserState, classify_user_state
from shiki_recsys.inference.warm_pipeline import build_warm_ranking
from shiki_recsys.model_artifacts import ArtifactInferenceConfig


@dataclass(frozen=True)
class RecommendationResult:
    """Хранит результат recommendation inference."""

    state: UserState
    recommendations: pd.DataFrame


def build_recommendations(
    *,
    user_id: int,
    interactions: pd.DataFrame,
    user_exists: bool,
    history_synced: bool,
    bundle: ModelBundle,
    artifact_config: ArtifactInferenceConfig,
    serving_config: RecommendationServingConfig,
) -> RecommendationResult:
    """
    Формирует рекомендации пользователя.

    Args:
        user_id: Идентификатор пользователя.
        interactions: Подготовленная локальная история пользователя.
        user_exists: Подтверждено ли существование пользователя.
        history_synced: Была ли история успешно синхронизирована.
        bundle: Загруженный набор моделей.
        artifact_config: Inference-параметры model artifact.
        serving_config: Параметры recommendation serving.

    Returns:
        Состояние пользователя и финальные рекомендации.
    """
    interaction_count = len(interactions)

    if user_exists and history_synced and interaction_count > 0:
        supported_positive_count = count_supported_positive_items(
            interactions,
            bundle.content_tfidf.item_features,
            relevance_threshold=artifact_config.positive_rating_threshold,
        )
        supports_personal_retriever = bundle.supports_personal_user(user_id)
    else:
        supported_positive_count = 0
        supports_personal_retriever = False

    state = classify_user_state(
        user_exists=user_exists,
        history_synced=history_synced,
        interaction_count=interaction_count,
        supported_positive_count=supported_positive_count,
        supports_personal_retriever=supports_personal_retriever,
        min_positive_items=serving_config.min_positive_items,
    )

    if state in {
        UserState.USER_NOT_FOUND,
        UserState.NOT_SYNCED,
    }:
        return RecommendationResult(
            state=state,
            recommendations=pd.DataFrame(
                {
                    "anime_id": pd.Series(dtype="int64"),
                    "rank": pd.Series(dtype="int32"),
                }
            ),
        )

    known_anime_ids = {int(anime_id) for anime_id in interactions["anime_id"]}

    if state == UserState.WARM:
        ranking = build_warm_ranking(
            user_id=user_id,
            known_anime_ids=known_anime_ids,
            candidate_count=artifact_config.retrieval_k,
            popularity=bundle.popularity,
            explicit_svd=bundle.explicit_svd,
            implicit_als=bundle.implicit_als,
            content_tfidf=bundle.content_tfidf,
            ranker=bundle.ranker,
        )
    else:
        ranking = build_cold_ranking(
            state=state,
            interactions=interactions,
            known_anime_ids=known_anime_ids,
            candidate_count=artifact_config.retrieval_k,
            relevance_threshold=artifact_config.positive_rating_threshold,
            max_positive_items=artifact_config.max_positive_items,
            popularity=bundle.popularity,
            content_tfidf=bundle.content_tfidf,
        )

    recommendations = ranking.head(serving_config.top_k).reset_index(drop=True)

    return RecommendationResult(
        state=state,
        recommendations=recommendations,
    )
