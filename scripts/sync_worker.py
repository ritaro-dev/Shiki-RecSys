import logging
import time
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from shiki_recsys.application.process_sync_job import (
    process_next_sync_job,
)
from shiki_recsys.config.settings import get_settings
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
from shiki_recsys.database.session import (
    create_database_engine,
    create_session_factory,
)
from shiki_recsys.integrations.shikimori.client import (
    ShikimoriClient,
)
from shiki_recsys.integrations.shikimori.rate_limiter import (
    ShikimoriRateLimiter,
)
from shiki_recsys.sync_jobs import SyncJobErrorCode

logger = logging.getLogger(__name__)


def run_worker(
    *,
    session_factory: sessionmaker[Session],
    client: ShikimoriClient,
    sync_job_repository: SyncJobRepository,
    user_repository: UserRepository,
    anime_repository: AnimeRepository,
    rates_repository: UserRateSVDRepository,
    poll_interval_seconds: float,
    stale_after_seconds: float,
) -> None:
    """
    Continuously process pending synchronization jobs.

    Args:
        session_factory: Database session factory.
        client: Shikimori API client.
        sync_job_repository: Repository for synchronization jobs.
        user_repository: Repository for persisted users.
        anime_repository: Repository for anime catalog data.
        rates_repository: Repository for persisted interactions.
        poll_interval_seconds: Delay when the queue is empty.
        stale_after_seconds: Maximum running time before a job is considered stale.

    Raises:
        ValueError: If a worker timing interval is not positive.
    """
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive.")

    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive.")

    while True:
        now = datetime.now(UTC)

        with session_factory() as session, session.begin():
            recovered_count = sync_job_repository.fail_stale_running(
                session=session,
                stale_before=now - timedelta(seconds=stale_after_seconds),
                finished_at=now,
                error_code=SyncJobErrorCode.WORKER_TIMEOUT.value,
                error_message="Synchronization worker did not finish the job in time.",
            )

        if recovered_count:
            logger.warning(
                "Marked %s stale synchronization job(s) as failed.",
                recovered_count,
            )

        with session_factory() as session:
            job = process_next_sync_job(
                session=session,
                client=client,
                sync_job_repository=sync_job_repository,
                user_repository=user_repository,
                anime_repository=anime_repository,
                rates_repository=rates_repository,
            )

        if job is None:
            time.sleep(poll_interval_seconds)
            continue

        logger.info(
            "Synchronization job %s finished with status %s.",
            job.id,
            job.status,
        )


def main() -> None:
    """Run the synchronization worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = get_settings()

    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    limiter = ShikimoriRateLimiter(
        min_interval_seconds=settings.shikimori_min_interval_seconds,
    )

    client = ShikimoriClient(
        graphql_url=settings.shikimori_graphql_url,
        user_agent=settings.shikimori_user_agent,
        limiter=limiter,
        timeout_seconds=settings.shikimori_timeout_seconds,
        max_retries=settings.shikimori_max_retries,
    )

    sync_job_repository = SyncJobRepository()
    user_repository = UserRepository()
    anime_repository = AnimeRepository()
    rates_repository = UserRateSVDRepository()

    logger.info("Synchronization worker started.")

    try:
        run_worker(
            session_factory=session_factory,
            client=client,
            sync_job_repository=sync_job_repository,
            user_repository=user_repository,
            anime_repository=anime_repository,
            rates_repository=rates_repository,
            poll_interval_seconds=settings.sync_worker_poll_interval_seconds,
            stale_after_seconds=settings.sync_job_stale_after_seconds,
        )
    except KeyboardInterrupt:
        logger.info("Synchronization worker stopped.")
    finally:
        client.close()
        engine.dispose()


if __name__ == "__main__":
    main()
