from collections.abc import Generator

from fastapi import Request
from sqlalchemy.orm import Session

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


def get_session(
    request: Request,
) -> Generator[Session, None, None]:
    """
    Provide a database session for one HTTP request.

    Args:
        request: Current FastAPI request.

    Yields:
        Database session scoped to the request.
    """

    session_factory = request.app.state.session_factory

    with session_factory() as session:
        yield session


def get_user_repository() -> UserRepository:
    return UserRepository()


def get_anime_repository() -> AnimeRepository:
    return AnimeRepository()


def get_rates_repository() -> UserRateSVDRepository:
    return UserRateSVDRepository()


def get_sync_job_repository() -> SyncJobRepository:
    """Return a synchronization job repository."""
    return SyncJobRepository()


def get_inference_state(
    request: Request,
) -> InferenceState:
    """
    Return the inference state loaded at application startup.

    Args:
        request: Current FastAPI request.

    Returns:
        Long-lived recommendation inference state.
    """
    return request.app.state.inference
