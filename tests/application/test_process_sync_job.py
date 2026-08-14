from contextlib import nullcontext
from datetime import UTC
from types import SimpleNamespace

import shiki_recsys.application.process_sync_job as process_sync_job_module
from shiki_recsys.sync_jobs import SyncJobErrorCode, SyncJobStatus

USER_ID = 315632


class FakeSession:
    def begin(self):
        return nullcontext()


class FakeSyncJobRepository:
    def __init__(self, *, job=None):
        self.job = job
        self.started_at = None
        self.finished_at = None
        self.error_code = None
        self.error_message = None

    def claim_next_pending(
        self,
        session,
        *,
        started_at,
    ):
        self.started_at = started_at

        if self.job is not None:
            self.job.status = SyncJobStatus.RUNNING.value

        return self.job

    def mark_completed(
        self,
        job,
        *,
        finished_at,
    ):
        job.status = SyncJobStatus.COMPLETED.value
        self.finished_at = finished_at

    def mark_failed(
        self,
        job,
        *,
        finished_at,
        error_code,
        error_message,
    ):
        job.status = SyncJobStatus.FAILED.value
        self.finished_at = finished_at
        self.error_code = error_code
        self.error_message = error_message


def make_job():
    return SimpleNamespace(
        id=1,
        user_id=USER_ID,
        status=SyncJobStatus.PENDING.value,
    )


def test_process_next_sync_job_returns_none_for_empty_queue(
    monkeypatch,
):
    repository = FakeSyncJobRepository()
    sync_called = False

    def fake_sync_user(**kwargs):
        nonlocal sync_called
        sync_called = True

    monkeypatch.setattr(
        process_sync_job_module,
        "sync_user",
        fake_sync_user,
    )

    result = process_sync_job_module.process_next_sync_job(
        session=FakeSession(),
        client=object(),
        sync_job_repository=repository,
        user_repository=object(),
        anime_repository=object(),
        rates_repository=object(),
    )

    assert result is None
    assert sync_called is False


def test_process_next_sync_job_completes_job(monkeypatch):
    job = make_job()
    repository = FakeSyncJobRepository(job=job)
    captured = {}

    def fake_sync_user(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        process_sync_job_module,
        "sync_user",
        fake_sync_user,
    )

    result = process_sync_job_module.process_next_sync_job(
        session=FakeSession(),
        client=object(),
        sync_job_repository=repository,
        user_repository=object(),
        anime_repository=object(),
        rates_repository=object(),
    )

    assert result is job
    assert job.status == SyncJobStatus.COMPLETED.value
    assert captured["user_id"] == USER_ID
    assert repository.started_at.tzinfo is UTC
    assert repository.finished_at.tzinfo is UTC


def test_process_next_sync_job_marks_failed_job(monkeypatch):
    job = make_job()
    repository = FakeSyncJobRepository(job=job)

    def fake_sync_user(**kwargs):
        raise RuntimeError("sync failed")

    monkeypatch.setattr(
        process_sync_job_module,
        "sync_user",
        fake_sync_user,
    )

    result = process_sync_job_module.process_next_sync_job(
        session=FakeSession(),
        client=object(),
        sync_job_repository=repository,
        user_repository=object(),
        anime_repository=object(),
        rates_repository=object(),
    )

    assert result is job
    assert job.status == SyncJobStatus.FAILED.value
    assert repository.error_code == SyncJobErrorCode.SYNC_FAILED.value
    assert repository.error_message == "sync failed"
    assert repository.finished_at.tzinfo is UTC
