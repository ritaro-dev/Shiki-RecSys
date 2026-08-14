import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from shiki_recsys.database.models.sync_job import SyncJob
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
from shiki_recsys.integrations.shikimori.client import ShikimoriClient
from shiki_recsys.sync_jobs import SyncJobErrorCode

from .sync_user import sync_user

logger = logging.getLogger(__name__)


def process_next_sync_job(
    *,
    session: Session,
    client: ShikimoriClient,
    sync_job_repository: SyncJobRepository,
    user_repository: UserRepository,
    anime_repository: AnimeRepository,
    rates_repository: UserRateSVDRepository,
) -> SyncJob | None:
    """
    Process the oldest pending synchronization job.

    Args:
        session: Database session.
        client: Shikimori API client.
        sync_job_repository: Repository for synchronization jobs.
        user_repository: Repository for persisted users.
        anime_repository: Repository for anime catalog data.
        rates_repository: Repository for persisted interactions.

    Returns:
        Processed job or None when the queue is empty.
    """
    with session.begin():
        job = sync_job_repository.claim_next_pending(
            session=session,
            started_at=datetime.now(UTC),
        )

    if job is None:
        return None

    try:
        sync_user(
            session=session,
            client=client,
            user_repository=user_repository,
            anime_repository=anime_repository,
            rates_repository=rates_repository,
            user_id=job.user_id,
        )
    except Exception as exc:
        logger.exception(
            "Synchronization job %s failed for user %s.",
            job.id,
            job.user_id,
        )

        with session.begin():
            sync_job_repository.mark_failed(
                job,
                finished_at=datetime.now(UTC),
                error_code=SyncJobErrorCode.SYNC_FAILED.value,
                error_message=str(exc),
            )

        return job

    with session.begin():
        sync_job_repository.mark_completed(
            job,
            finished_at=datetime.now(UTC),
        )

    return job
