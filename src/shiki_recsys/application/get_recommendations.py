from sqlalchemy.orm import Session

from shiki_recsys.config.inference import RecommendationServingConfig
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
from shiki_recsys.model_artifacts import ArtifactInferenceConfig
from shiki_recsys.preprocessing.interactions import prepare_interactions


def get_recommendations(
    *,
    session: Session,
    user_repository: UserRepository,
    rates_repository: UserRateSVDRepository,
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

    return build_recommendations(
        user_id=user_id,
        interactions=interactions,
        user_exists=user_exists,
        history_synced=history_synced,
        bundle=bundle,
        artifact_config=artifact_config,
        serving_config=serving_config,
    )
