from typing import Annotated

from fastapi import APIRouter, Depends, Path, status
from sqlalchemy.orm import Session

from shiki_recsys.api.dependencies import (
    get_anime_repository,
    get_inference_state,
    get_rates_repository,
    get_session,
    get_sync_job_repository,
    get_user_repository,
)
from shiki_recsys.api.schemas.recommendations import (
    RecommendationItemResponse,
    RecommendationsResponse,
)
from shiki_recsys.api.schemas.sync_jobs import SyncJobResponse
from shiki_recsys.api.schemas.users import (
    CreateUserRequest,
    UserResponse,
)
from shiki_recsys.application.add_user import add_user
from shiki_recsys.application.enqueue_sync_job import enqueue_sync_job
from shiki_recsys.application.get_recommendations import (
    get_recommendations,
)
from shiki_recsys.application.get_sync_job import (
    get_latest_sync_job,
)
from shiki_recsys.database.repositories.anime_repository import (
    AnimeRepository,
)
from shiki_recsys.database.repositories.sync_job_repository import (
    SyncJobRepository,
)
from shiki_recsys.database.repositories.user_rate_svd_repository import (
    UserRateSVDRepository,
)
from shiki_recsys.database.repositories.user_repository import (
    UserRepository,
)
from shiki_recsys.inference.runtime import InferenceState

router = APIRouter(
    prefix="/users",
    tags=["users"],
)


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    payload: CreateUserRequest,
    session: Annotated[
        Session,
        Depends(get_session),
    ],
    user_repository: Annotated[
        UserRepository,
        Depends(get_user_repository),
    ],
) -> UserResponse:
    """
    Register a Shikimori user in the recommendation system.

    The user's history is synchronized separately.
    """

    user = add_user(
        session=session,
        user_repository=user_repository,
        user_id=payload.user_id,
    )

    return UserResponse.model_validate(user)


@router.post(
    "/{user_id}/sync",
    response_model=SyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_user_sync(
    user_id: Annotated[
        int,
        Path(
            gt=0,
            description="Shikimori user ID.",
        ),
    ],
    session: Annotated[
        Session,
        Depends(get_session),
    ],
    user_repository: Annotated[
        UserRepository,
        Depends(get_user_repository),
    ],
    sync_job_repository: Annotated[
        SyncJobRepository,
        Depends(get_sync_job_repository),
    ],
) -> SyncJobResponse:
    """
    Enqueue synchronization for a registered user.

    Args:
        user_id: Shikimori user ID.
        session: Database session.
        user_repository: Repository for persisted users.
        sync_job_repository: Repository for synchronization jobs.

    Returns:
        Existing or newly created active synchronization job.
    """
    job = enqueue_sync_job(
        session=session,
        user_repository=user_repository,
        sync_job_repository=sync_job_repository,
        user_id=user_id,
    )

    return SyncJobResponse.model_validate(job)


@router.get(
    "/{user_id}/sync",
    response_model=SyncJobResponse,
)
def get_user_sync(
    user_id: Annotated[
        int,
        Path(
            gt=0,
            description="Shikimori user ID.",
        ),
    ],
    session: Annotated[
        Session,
        Depends(get_session),
    ],
    user_repository: Annotated[
        UserRepository,
        Depends(get_user_repository),
    ],
    sync_job_repository: Annotated[
        SyncJobRepository,
        Depends(get_sync_job_repository),
    ],
) -> SyncJobResponse:
    """
    Return the latest synchronization job for a user.

    Args:
        user_id: Shikimori user ID.
        session: Database session.
        user_repository: Repository for persisted users.
        sync_job_repository: Repository for synchronization jobs.

    Returns:
        Most recent synchronization job.
    """
    job = get_latest_sync_job(
        session=session,
        user_repository=user_repository,
        sync_job_repository=sync_job_repository,
        user_id=user_id,
    )

    return SyncJobResponse.model_validate(job)


@router.get(
    "/{user_id}/recommendations",
    response_model=RecommendationsResponse,
)
def recommend_anime(
    user_id: Annotated[
        int,
        Path(
            gt=0,
            description="Shikimori user ID.",
        ),
    ],
    session: Annotated[
        Session,
        Depends(get_session),
    ],
    inference: Annotated[
        InferenceState,
        Depends(get_inference_state),
    ],
    user_repository: Annotated[
        UserRepository,
        Depends(get_user_repository),
    ],
    anime_repository: Annotated[
        AnimeRepository,
        Depends(get_anime_repository),
    ],
    rates_repository: Annotated[
        UserRateSVDRepository,
        Depends(get_rates_repository),
    ],
) -> RecommendationsResponse:
    """
    Return ranked anime recommendations for a user.

    Args:
        user_id: Shikimori user ID.
        session: Database session.
        inference: Loaded recommendation inference state.
        user_repository: Repository for persisted users.
        anime_repository: Repository for anime catalog metadata.
        rates_repository: Repository for persisted interactions.

    Returns:
        Ranked anime recommendations and the resolved user state.
    """
    result = get_recommendations(
        session=session,
        user_repository=user_repository,
        anime_repository=anime_repository,
        rates_repository=rates_repository,
        user_id=user_id,
        bundle=inference.bundle,
        artifact_config=inference.metadata.inference,
        serving_config=inference.serving_config,
    )

    recommendations = [
        RecommendationItemResponse(
            anime_id=int(row.anime_id),
            display_name=str(row.display_name),
            rank=int(row.rank),
        )
        for row in result.recommendations.itertuples(index=False)
    ]

    return RecommendationsResponse(
        user_id=user_id,
        state=result.state,
        recommendations=recommendations,
    )
