from sqlalchemy.orm import Session

from shiki_recsys.database.models.sync_job import SyncJob
from shiki_recsys.database.repositories.sync_job_repository import (
    SyncJobRepository,
)
from shiki_recsys.database.repositories.user_repository import (
    UserRepository,
)

from .exceptions import SyncJobNotFoundError, UserNotFoundError


def get_latest_sync_job(
    *,
    session: Session,
    user_repository: UserRepository,
    sync_job_repository: SyncJobRepository,
    user_id: int,
) -> SyncJob:
    """
    Return the most recent synchronization job for a user.

    Args:
        session: Database session.
        user_repository: Repository for persisted users.
        sync_job_repository: Repository for synchronization jobs.
        user_id: Shikimori user ID.

    Returns:
        Most recent synchronization job.

    Raises:
        ValueError: If the user ID is not positive.
        UserNotFoundError: If the user is not registered.
        SyncJobNotFoundError: If no synchronization job exists.
    """
    if user_id <= 0:
        raise ValueError("user_id должен быть больше 0.")

    user = user_repository.get_by_id(
        session=session,
        user_id=user_id,
    )

    if user is None:
        raise UserNotFoundError(f"Пользователь {user_id} не найден.")

    job = sync_job_repository.get_latest_for_user(
        session=session,
        user_id=user_id,
    )

    if job is None:
        raise SyncJobNotFoundError(f"Synchronization job for user {user_id} not found.")

    return job
