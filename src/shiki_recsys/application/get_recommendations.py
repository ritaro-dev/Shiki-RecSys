from sqlalchemy.orm import Session

from shiki_recsys.application.exceptions import UserNotSyncedError
from shiki_recsys.config.inference import RecommendationServingConfig
from shiki_recsys.database.repositories.anime_repository import (
    AnimeRepository,
)
from shiki_recsys.database.repositories.user_rate_svd_repository import (
    UserRateSVDRepository,
)
from shiki_recsys.database.repositories.user_repository import (
    UserRepository,
)
from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.inference.recommendation_service import (
    RecommendationResult,
    build_recommendations,
)
from shiki_recsys.inference.user_state import UserState
from shiki_recsys.model_artifacts import ArtifactInferenceConfig
from shiki_recsys.preprocessing.interactions import prepare_interactions


def get_recommendations(
    *,
    session: Session,
    user_repository: UserRepository,
    rates_repository: UserRateSVDRepository,
    anime_repository: AnimeRepository,
    user_id: int,
    bundle: ModelBundle,
    artifact_config: ArtifactInferenceConfig,
    serving_config: RecommendationServingConfig,
) -> RecommendationResult:
    """
    Build recommendations from the user's persisted history.

    Args:
        session: Database session.
        user_repository: Repository for persisted users.
        rates_repository: Repository for persisted interactions.
        anime_repository: Repository for anime catalog metadata.
        user_id: Shikimori user ID.
        bundle: Loaded inference model bundle.
        artifact_config: Artifact-bound inference configuration.
        serving_config: Recommendation serving configuration.

    Returns:
        Recommendation result with the resolved user state.
    """
    user = user_repository.get_by_id(
        session=session,
        user_id=user_id,
    )

    history_synced = user is not None and user.last_synced_at is not None
    user_exists = True if history_synced else None

    if history_synced:
        rows = rates_repository.get_by_user_id(
            session=session,
            user_id=user_id,
        )
    else:
        rows = []

    interactions = prepare_interactions(rows)

    result = build_recommendations(
        user_id=user_id,
        interactions=interactions,
        user_exists=user_exists,
        history_synced=history_synced,
        bundle=bundle,
        artifact_config=artifact_config,
        serving_config=serving_config,
    )

    if result.state == UserState.NOT_SYNCED:
        raise UserNotSyncedError(f"User {user_id} has not been synchronized.")

    if result.recommendations.empty:
        return result

    anime_ids = [int(anime_id) for anime_id in result.recommendations["anime_id"]]

    title_rows = anime_repository.get_titles_by_ids(
        session=session,
        anime_ids=anime_ids,
    )

    display_names = {
        int(row["id"]): str(row["russian_name"] or row["name"]) for row in title_rows
    }

    recommendations = result.recommendations.copy()
    recommendations["display_name"] = recommendations["anime_id"].map(display_names)

    return RecommendationResult(
        state=result.state,
        recommendations=recommendations,
    )
