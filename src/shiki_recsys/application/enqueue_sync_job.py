from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shiki_recsys.database.models.sync_job import SyncJob
from shiki_recsys.database.repositories.sync_job_repository import (
    SyncJobRepository,
)
from shiki_recsys.database.repositories.user_repository import (
    UserRepository,
)

from .exceptions import UserNotFoundError


def enqueue_sync_job(
    *,
    session: Session,
    user_repository: UserRepository,
    sync_job_repository: SyncJobRepository,
    user_id: int,
) -> SyncJob:
    """
    Enqueue user synchronization or reuse the active job.

    Args:
        session: Database session.
        user_repository: Repository for persisted users.
        sync_job_repository: Repository for synchronization jobs.
        user_id: Shikimori user ID.

    Returns:
        Existing or newly created active synchronization job.

    Raises:
        ValueError: If the user ID is not positive.
        UserNotFoundError: If the user is not registered locally.
    """
    if user_id <= 0:
        raise ValueError("user_id должен быть больше 0.")

    try:
        with session.begin():
            user = user_repository.get_by_id(
                session=session,
                user_id=user_id,
            )

            if user is None:
                raise UserNotFoundError(f"Пользователь {user_id} не найден.")

            active_job = sync_job_repository.get_active_for_user(
                session=session,
                user_id=user_id,
            )

            if active_job is not None:
                return active_job

            job = sync_job_repository.add_pending(
                session=session,
                user_id=user_id,
            )
            session.flush()

        return job

    except IntegrityError:
        with session.begin():
            active_job = sync_job_repository.get_active_for_user(
                session=session,
                user_id=user_id,
            )

            if active_job is None:
                raise

            return active_job
