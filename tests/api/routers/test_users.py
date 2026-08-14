from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

import shiki_recsys.api.routers.users as users_router_module
import shiki_recsys.application.sync_user as sync_user_module
from shiki_recsys.api.dependencies import (
    get_anime_repository,
    get_inference_state,
    get_rates_repository,
    get_session,
    get_shikimori_client,
    get_user_repository,
)
from shiki_recsys.api.exception_handlers import (
    register_exception_handlers,
)
from shiki_recsys.api.routers.users import (
    router as users_router,
)
from shiki_recsys.application.exceptions import UserNotSyncedError
from shiki_recsys.inference.recommendation_service import RecommendationResult
from shiki_recsys.inference.user_state import UserState

USER_ID = 315632
CREATED_AT = datetime(
    2026,
    8,
    3,
    10,
    0,
    tzinfo=UTC,
)


def make_user(
    *,
    user_id: int = USER_ID,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=user_id,
        created_at=CREATED_AT,
        last_synced_at=None,
    )


class FakeSession:
    def begin(self):
        return nullcontext()


class FakeUserRepository:
    def __init__(
        self,
        *,
        user: SimpleNamespace | None = None,
    ):
        self.user = user
        self.added_user_id: int | None = None

    def get_by_id(
        self,
        session,
        *,
        user_id: int,
    ):
        return self.user

    def add(
        self,
        session,
        *,
        user_id: int,
    ):
        self.added_user_id = user_id
        self.user = make_user(user_id=user_id)

        return self.user

    def mark_synced(
        self,
        user,
        *,
        synced_at: datetime,
    ) -> None:
        user.last_synced_at = synced_at


class FakeAnimeRepository:
    def get_all_ids(self, session):
        return {1}


class FakeRatesRepository:
    def __init__(self):
        self.deleted_user_id: int | None = None
        self.saved_rows: list[dict] | None = None

    def delete_by_user_id(
        self,
        session,
        *,
        user_id: int,
    ) -> None:
        self.deleted_user_id = user_id

    def upsert_many(
        self,
        session,
        *,
        rate_rows: list[dict],
    ) -> int:
        self.saved_rows = rate_rows
        return len(rate_rows)


def create_test_client(
    *,
    user_repository: FakeUserRepository,
    anime_repository: FakeAnimeRepository | None = None,
    rates_repository: FakeRatesRepository | None = None,
    inference_state=None,
) -> TestClient:
    app = FastAPI()

    register_exception_handlers(app)
    app.include_router(users_router)

    app.dependency_overrides[get_session] = lambda: FakeSession()
    app.dependency_overrides[get_shikimori_client] = lambda: object()
    app.dependency_overrides[get_user_repository] = lambda: user_repository
    app.dependency_overrides[get_anime_repository] = lambda: (
        anime_repository or FakeAnimeRepository()
    )
    app.dependency_overrides[get_rates_repository] = lambda: (
        rates_repository or FakeRatesRepository()
    )

    if inference_state is not None:
        app.dependency_overrides[get_inference_state] = lambda: inference_state

    return TestClient(app)


def test_create_user_returns_201():
    repository = FakeUserRepository()
    client = create_test_client(
        user_repository=repository,
    )

    response = client.post(
        "/users",
        json={
            "user_id": USER_ID,
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": USER_ID,
        "created_at": "2026-08-03T10:00:00Z",
        "last_synced_at": None,
    }

    assert repository.added_user_id == USER_ID


def test_create_existing_user_returns_409():
    repository = FakeUserRepository(
        user=make_user(),
    )
    client = create_test_client(
        user_repository=repository,
    )

    response = client.post(
        "/users",
        json={
            "user_id": USER_ID,
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": (f"Пользователь {USER_ID} уже добавлен."),
    }

    assert repository.added_user_id is None


def test_create_user_rejects_non_positive_id():
    repository = FakeUserRepository()
    client = create_test_client(
        user_repository=repository,
    )

    response = client.post(
        "/users",
        json={
            "user_id": 0,
        },
    )

    assert response.status_code == 422
    assert repository.added_user_id is None


def test_sync_user_returns_200(monkeypatch):
    history = [
        {
            "score": 8,
            "status": "completed",
            "updatedAt": "2026-08-03T10:00:00Z",
            "anime": {
                "id": "1",
            },
        },
    ]

    monkeypatch.setattr(
        sync_user_module,
        "fetch_user_history",
        lambda *, client, user_id: history,
    )

    user_repository = FakeUserRepository(
        user=make_user(),
    )
    anime_repository = FakeAnimeRepository()
    rates_repository = FakeRatesRepository()

    client = create_test_client(
        user_repository=user_repository,
        anime_repository=anime_repository,
        rates_repository=rates_repository,
    )

    response = client.post(
        f"/users/{USER_ID}/sync",
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["id"] == USER_ID
    assert response_data["created_at"] == ("2026-08-03T10:00:00Z")
    assert response_data["last_synced_at"] is not None

    assert rates_repository.deleted_user_id == USER_ID
    assert rates_repository.saved_rows is not None
    assert len(rates_repository.saved_rows) == 1
    assert rates_repository.saved_rows[0]["anime_id"] == 1
    assert rates_repository.saved_rows[0]["rating"] == 8


def test_sync_missing_user_returns_404():
    repository = FakeUserRepository()
    client = create_test_client(
        user_repository=repository,
    )

    response = client.post(
        "/users/2147483647/sync",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": ("Пользователь 2147483647 не найден."),
    }


def test_sync_rejects_non_positive_user_id():
    repository = FakeUserRepository()
    client = create_test_client(
        user_repository=repository,
    )

    response = client.post(
        "/users/0/sync",
    )

    assert response.status_code == 422


def test_get_recommendations_returns_ranked_items(monkeypatch):
    bundle = object()
    artifact_config = object()
    serving_config = object()

    inference_state = SimpleNamespace(
        bundle=bundle,
        metadata=SimpleNamespace(
            inference=artifact_config,
        ),
        serving_config=serving_config,
    )

    captured = {}

    def fake_get_recommendations(**kwargs):
        captured.update(kwargs)

        return RecommendationResult(
            state=UserState.WARM,
            recommendations=pd.DataFrame(
                {
                    "anime_id": [1, 2],
                    "display_name": [
                        "Стальной алхимик",
                        "Monster",
                    ],
                    "rank": [1, 2],
                }
            ),
        )

    monkeypatch.setattr(
        users_router_module,
        "get_recommendations",
        fake_get_recommendations,
    )

    user_repository = FakeUserRepository(
        user=make_user(),
    )

    client = create_test_client(
        user_repository=user_repository,
        inference_state=inference_state,
    )

    response = client.get(
        f"/users/{USER_ID}/recommendations",
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": USER_ID,
        "state": "warm",
        "recommendations": [
            {
                "anime_id": 1,
                "display_name": "Стальной алхимик",
                "rank": 1,
            },
            {
                "anime_id": 2,
                "display_name": "Monster",
                "rank": 2,
            },
        ],
    }

    assert captured["user_id"] == USER_ID
    assert captured["bundle"] is bundle
    assert captured["artifact_config"] is artifact_config
    assert captured["serving_config"] is serving_config


def test_get_recommendations_returns_409_for_unsynced_user(
    monkeypatch,
):
    inference_state = SimpleNamespace(
        bundle=object(),
        metadata=SimpleNamespace(
            inference=object(),
        ),
        serving_config=object(),
    )

    def fake_get_recommendations(**kwargs):
        raise UserNotSyncedError(f"User {USER_ID} has not been synchronized.")

    monkeypatch.setattr(
        users_router_module,
        "get_recommendations",
        fake_get_recommendations,
    )

    client = create_test_client(
        user_repository=FakeUserRepository(),
        inference_state=inference_state,
    )

    response = client.get(
        f"/users/{USER_ID}/recommendations",
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": f"User {USER_ID} has not been synchronized.",
    }


def test_get_recommendations_rejects_non_positive_user_id():
    inference_state = SimpleNamespace(
        bundle=object(),
        metadata=SimpleNamespace(
            inference=object(),
        ),
        serving_config=object(),
    )

    client = create_test_client(
        user_repository=FakeUserRepository(),
        inference_state=inference_state,
    )

    response = client.get(
        "/users/0/recommendations",
    )

    assert response.status_code == 422
