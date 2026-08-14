from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

from shiki_recsys.application.enqueue_sync_job import (
    enqueue_sync_job,
)
from shiki_recsys.application.exceptions import UserNotFoundError
from shiki_recsys.sync_jobs import SyncJobStatus

USER_ID = 315632


class FakeSession:
    def __init__(self, *, flush_error=None):
        self.flush_error = flush_error
        self.flush_calls = 0

    def begin(self):
        return nullcontext()

    def flush(self):
        self.flush_calls += 1

        if self.flush_error is not None:
            error = self.flush_error
            self.flush_error = None
            raise error


class FakeUserRepository:
    def __init__(self, *, user=None):
        self.user = user

    def get_by_id(self, session, *, user_id):
        return self.user


class FakeSyncJobRepository:
    def __init__(self, *, active_jobs=None):
        self.active_jobs = list(active_jobs or [])
        self.added_user_id = None

    def get_active_for_user(
        self,
        session,
        *,
        user_id,
    ):
        if not self.active_jobs:
            return None

        return self.active_jobs.pop(0)

    def add_pending(
        self,
        session,
        *,
        user_id,
    ):
        self.added_user_id = user_id

        return SimpleNamespace(
            id=1,
            user_id=user_id,
            status=SyncJobStatus.PENDING.value,
        )


def test_enqueue_sync_job_creates_pending_job():
    session = FakeSession()
    repository = FakeSyncJobRepository()

    job = enqueue_sync_job(
        session=session,
        user_repository=FakeUserRepository(
            user=SimpleNamespace(id=USER_ID),
        ),
        sync_job_repository=repository,
        user_id=USER_ID,
    )

    assert job.user_id == USER_ID
    assert job.status == SyncJobStatus.PENDING.value
    assert repository.added_user_id == USER_ID
    assert session.flush_calls == 1


def test_enqueue_sync_job_reuses_active_job():
    active_job = SimpleNamespace(
        id=10,
        user_id=USER_ID,
        status=SyncJobStatus.RUNNING.value,
    )
    session = FakeSession()
    repository = FakeSyncJobRepository(
        active_jobs=[active_job],
    )

    job = enqueue_sync_job(
        session=session,
        user_repository=FakeUserRepository(
            user=SimpleNamespace(id=USER_ID),
        ),
        sync_job_repository=repository,
        user_id=USER_ID,
    )

    assert job is active_job
    assert repository.added_user_id is None
    assert session.flush_calls == 0


def test_enqueue_sync_job_rejects_missing_user():
    repository = FakeSyncJobRepository()

    with pytest.raises(
        UserNotFoundError,
        match=f"Пользователь {USER_ID} не найден.",
    ):
        enqueue_sync_job(
            session=FakeSession(),
            user_repository=FakeUserRepository(),
            sync_job_repository=repository,
            user_id=USER_ID,
        )

    assert repository.added_user_id is None


def test_enqueue_sync_job_recovers_from_concurrent_insert():
    active_job = SimpleNamespace(
        id=10,
        user_id=USER_ID,
        status=SyncJobStatus.PENDING.value,
    )
    session = FakeSession(
        flush_error=IntegrityError(
            "INSERT INTO sync_jobs",
            {},
            Exception("duplicate active sync job"),
        )
    )
    repository = FakeSyncJobRepository(
        active_jobs=[
            None,
            active_job,
        ],
    )

    job = enqueue_sync_job(
        session=session,
        user_repository=FakeUserRepository(
            user=SimpleNamespace(id=USER_ID),
        ),
        sync_job_repository=repository,
        user_id=USER_ID,
    )

    assert job is active_job
    assert repository.added_user_id == USER_ID
    assert session.flush_calls == 1
