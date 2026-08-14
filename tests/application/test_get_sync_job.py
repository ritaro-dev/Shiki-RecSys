from types import SimpleNamespace

import pytest

from shiki_recsys.application.exceptions import (
    SyncJobNotFoundError,
    UserNotFoundError,
)
from shiki_recsys.application.get_sync_job import (
    get_latest_sync_job,
)

USER_ID = 315632


class FakeUserRepository:
    def __init__(self, *, user=None):
        self.user = user

    def get_by_id(self, session, *, user_id):
        return self.user


class FakeSyncJobRepository:
    def __init__(self, *, job=None):
        self.job = job

    def get_latest_for_user(self, session, *, user_id):
        return self.job


def test_get_latest_sync_job_returns_job():
    job = SimpleNamespace(id=42, user_id=USER_ID)

    result = get_latest_sync_job(
        session=object(),
        user_repository=FakeUserRepository(
            user=SimpleNamespace(id=USER_ID),
        ),
        sync_job_repository=FakeSyncJobRepository(job=job),
        user_id=USER_ID,
    )

    assert result is job


def test_get_latest_sync_job_rejects_missing_user():
    with pytest.raises(
        UserNotFoundError,
        match=f"Пользователь {USER_ID} не найден.",
    ):
        get_latest_sync_job(
            session=object(),
            user_repository=FakeUserRepository(),
            sync_job_repository=FakeSyncJobRepository(),
            user_id=USER_ID,
        )


def test_get_latest_sync_job_rejects_missing_job():
    with pytest.raises(
        SyncJobNotFoundError,
        match=f"Synchronization job for user {USER_ID} not found.",
    ):
        get_latest_sync_job(
            session=object(),
            user_repository=FakeUserRepository(
                user=SimpleNamespace(id=USER_ID),
            ),
            sync_job_repository=FakeSyncJobRepository(),
            user_id=USER_ID,
        )
