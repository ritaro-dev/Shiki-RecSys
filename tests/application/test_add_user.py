from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from shiki_recsys.application.add_user import add_user
from shiki_recsys.application.exceptions import (
    UserAlreadyExistsError,
)

USER_ID = 315632


class FakeSession:
    def begin(self):
        return nullcontext()


class FakeUserRepository:
    def __init__(self, *, existing_user=None):
        self.existing_user = existing_user
        self.added_user_id: int | None = None

    def get_by_id(
        self,
        session,
        *,
        user_id: int,
    ):
        return self.existing_user

    def add(
        self,
        session,
        *,
        user_id: int,
    ):
        self.added_user_id = user_id

        return SimpleNamespace(
            id=user_id,
            created_at=None,
            last_synced_at=None,
        )


def test_add_user_creates_new_user():
    repository = FakeUserRepository()

    user = add_user(
        session=FakeSession(),
        user_repository=repository,
        user_id=USER_ID,
    )

    assert user.id == USER_ID
    assert repository.added_user_id == USER_ID


def test_add_user_rejects_existing_user():
    existing_user = SimpleNamespace(
        id=USER_ID,
    )

    repository = FakeUserRepository(
        existing_user=existing_user,
    )

    with pytest.raises(
        UserAlreadyExistsError,
        match=f"Пользователь {USER_ID} уже добавлен.",
    ):
        add_user(
            session=FakeSession(),
            user_repository=repository,
            user_id=USER_ID,
        )

    assert repository.added_user_id is None


@pytest.mark.parametrize(
    "user_id",
    [0, -1],
)
def test_add_user_rejects_non_positive_id(
    user_id: int,
):
    repository = FakeUserRepository()

    with pytest.raises(
        ValueError,
        match="user_id должен быть больше 0",
    ):
        add_user(
            session=FakeSession(),
            user_repository=repository,
            user_id=user_id,
        )

    assert repository.added_user_id is None
