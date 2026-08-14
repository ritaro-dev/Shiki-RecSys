from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from shiki_recsys.database.models.sync_job import SyncJob
from shiki_recsys.sync_jobs import SyncJobStatus

ACTIVE_STATUSES = (
    SyncJobStatus.PENDING.value,
    SyncJobStatus.RUNNING.value,
)


class SyncJobRepository:
    """Provide persistence operations for synchronization jobs."""

    def get_active_for_user(
        self,
        session: Session,
        *,
        user_id: int,
    ) -> SyncJob | None:
        """
        Return the active synchronization job for a user.

        Args:
            session: Database session.
            user_id: Shikimori user ID.

        Returns:
            Pending or running job when one exists.
        """
        statement = select(SyncJob).where(
            SyncJob.user_id == user_id,
            SyncJob.status.in_(ACTIVE_STATUSES),
        )
        return session.scalar(statement)

    def get_latest_for_user(
        self,
        session: Session,
        *,
        user_id: int,
    ) -> SyncJob | None:
        """
        Return the most recent synchronization job for a user.

        Args:
            session: Database session.
            user_id: Shikimori user ID.

        Returns:
            Most recent job when one exists.
        """
        statement = (
            select(SyncJob)
            .where(SyncJob.user_id == user_id)
            .order_by(
                SyncJob.created_at.desc(),
                SyncJob.id.desc(),
            )
            .limit(1)
        )
        return session.scalar(statement)

    def add_pending(
        self,
        session: Session,
        *,
        user_id: int,
    ) -> SyncJob:
        """
        Add a pending synchronization job.

        Args:
            session: Database session.
            user_id: Shikimori user ID.

        Returns:
            Newly created synchronization job.
        """
        job = SyncJob(
            user_id=user_id,
            status=SyncJobStatus.PENDING.value,
        )
        session.add(job)
        return job

    def claim_next_pending(
        self,
        session: Session,
        *,
        started_at: datetime,
    ) -> SyncJob | None:
        """
        Claim the oldest pending job for processing.

        Args:
            session: Database session.
            started_at: Job processing start time.

        Returns:
            Claimed job when one is available.
        """
        statement = (
            select(SyncJob)
            .where(
                SyncJob.status == SyncJobStatus.PENDING.value,
            )
            .order_by(
                SyncJob.created_at,
                SyncJob.id,
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        )

        job = session.scalar(statement)

        if job is None:
            return None

        job.status = SyncJobStatus.RUNNING.value
        job.started_at = started_at

        return job

    def mark_completed(
        self,
        job: SyncJob,
        *,
        finished_at: datetime,
    ) -> None:
        """
        Mark a synchronization job as completed.

        Args:
            job: Synchronization job.
            finished_at: Job completion time.
        """
        job.status = SyncJobStatus.COMPLETED.value
        job.finished_at = finished_at
        job.error_code = None
        job.error_message = None

    def mark_failed(
        self,
        job: SyncJob,
        *,
        finished_at: datetime,
        error_code: str,
        error_message: str,
    ) -> None:
        """
        Mark a synchronization job as failed.

        Args:
            job: Synchronization job.
            finished_at: Job completion time.
            error_code: Stable failure code.
            error_message: Human-readable failure description.
        """
        job.status = SyncJobStatus.FAILED.value
        job.finished_at = finished_at
        job.error_code = error_code
        job.error_message = error_message

    def fail_stale_running(
        self,
        session: Session,
        *,
        stale_before: datetime,
        finished_at: datetime,
        error_code: str,
        error_message: str,
    ) -> int:
        """
        Mark stale running synchronization jobs as failed.

        Args:
            session: Database session.
            stale_before: Jobs started before this time are stale.
            finished_at: Failure time.
            error_code: Stable failure code.
            error_message: Human-readable failure description.

        Returns:
            Number of failed jobs.
        """
        statement = (
            update(SyncJob)
            .where(
                SyncJob.status == SyncJobStatus.RUNNING.value,
                SyncJob.started_at < stale_before,
            )
            .values(
                status=SyncJobStatus.FAILED.value,
                finished_at=finished_at,
                error_code=error_code,
                error_message=error_message,
            )
        )

        result = session.execute(statement)
        return result.rowcount
