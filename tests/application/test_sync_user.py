from contextlib import nullcontext
from types import SimpleNamespace

import shiki_recsys.application.sync_user as sync_user_module


class FakeSession:
    def begin(self):
        return nullcontext()


class FakeUserRepository:
    def __init__(self):
        self.user = SimpleNamespace(
            id=315632,
            last_synced_at=None,
        )

    def get_by_id(self, session, *, user_id):
        return self.user

    def mark_synced(self, user, *, synced_at):
        user.last_synced_at = synced_at


class FakeAnimeRepository:
    def get_all_ids(self, session):
        return {1, 2}


class FakeRatesRepository:
    def __init__(self):
        self.deleted_user_id = None
        self.saved_rows = None

    def delete_by_user_id(self, session, *, user_id):
        self.deleted_user_id = user_id

    def upsert_many(self, session, *, rate_rows):
        self.saved_rows = rate_rows
        return len(rate_rows)


def test_sync_user_replaces_history(monkeypatch):
    history = [
        {
            "score": 8,
            "status": "completed",
            "updatedAt": "2026-08-01T12:00:00Z",
            "anime": {"id": "1"},
        },
        {
            "score": 6,
            "status": "watching",
            "updatedAt": "2026-08-02T12:00:00Z",
            "anime": {"id": "2"},
        },
    ]

    monkeypatch.setattr(
        sync_user_module,
        "fetch_user_history",
        lambda *, client, user_id: history,
    )

    user_repository = FakeUserRepository()
    rates_repository = FakeRatesRepository()

    user = sync_user_module.sync_user(
        session=FakeSession(),
        client=object(),
        user_repository=user_repository,
        anime_repository=FakeAnimeRepository(),
        rates_repository=rates_repository,
        user_id=315632,
    )

    assert rates_repository.deleted_user_id == 315632
    assert rates_repository.saved_rows is not None
    assert len(rates_repository.saved_rows) == 2

    assert rates_repository.saved_rows[0]["anime_id"] == 1
    assert rates_repository.saved_rows[0]["rating"] == 8

    assert rates_repository.saved_rows[1]["anime_id"] == 2
    assert rates_repository.saved_rows[1]["rating"] == 6

    assert user.last_synced_at is not None
    assert user.last_synced_at.tzinfo is not None
