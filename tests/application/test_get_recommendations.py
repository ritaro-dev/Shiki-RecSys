from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd

import shiki_recsys.application.get_recommendations as get_recommendations_module
from shiki_recsys.inference.recommendation_service import (
    RecommendationResult,
)
from shiki_recsys.inference.user_state import UserState

USER_ID = 315632


class FakeUserRepository:
    def __init__(self, user=None):
        self.user = user

    def get_by_id(self, session, *, user_id):
        return self.user


class FakeRatesRepository:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.requested_user_id = None

    def get_by_user_id(self, session, *, user_id):
        self.requested_user_id = user_id
        return self.rows


class FakeAnimeRepository:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.requested_anime_ids = None

    def get_titles_by_ids(
        self,
        session,
        *,
        anime_ids,
    ):
        self.requested_anime_ids = anime_ids
        return self.rows


def _capture_build_recommendations(monkeypatch):
    """Capture arguments passed to recommendation inference."""
    captured = {}
    result = RecommendationResult(
        state=UserState.NOT_SYNCED,
        recommendations=pd.DataFrame(
            {
                "anime_id": pd.Series(dtype="int64"),
                "rank": pd.Series(dtype="int32"),
            }
        ),
    )

    def fake_build_recommendations(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(
        get_recommendations_module,
        "build_recommendations",
        fake_build_recommendations,
    )

    return captured, result


def test_get_recommendations_handles_missing_user(monkeypatch):
    captured, expected_result = _capture_build_recommendations(monkeypatch)
    rates_repository = FakeRatesRepository()

    result = get_recommendations_module.get_recommendations(
        session=object(),
        user_repository=FakeUserRepository(),
        rates_repository=rates_repository,
        anime_repository=FakeAnimeRepository(),
        user_id=USER_ID,
        bundle=object(),
        artifact_config=object(),
        serving_config=object(),
    )

    assert result is expected_result
    assert captured["user_exists"] is None
    assert captured["history_synced"] is False
    assert captured["interactions"].empty
    assert rates_repository.requested_user_id is None


def test_get_recommendations_handles_unsynced_user(monkeypatch):
    captured, expected_result = _capture_build_recommendations(monkeypatch)
    user = SimpleNamespace(
        id=USER_ID,
        last_synced_at=None,
    )
    rates_repository = FakeRatesRepository()

    result = get_recommendations_module.get_recommendations(
        session=object(),
        user_repository=FakeUserRepository(user),
        rates_repository=rates_repository,
        anime_repository=FakeAnimeRepository(),
        user_id=USER_ID,
        bundle=object(),
        artifact_config=object(),
        serving_config=object(),
    )

    assert result is expected_result
    assert captured["user_exists"] is None
    assert captured["history_synced"] is False
    assert captured["interactions"].empty
    assert rates_repository.requested_user_id is None


def test_get_recommendations_prepares_synced_history(monkeypatch):
    captured, expected_result = _capture_build_recommendations(monkeypatch)
    user = SimpleNamespace(
        id=USER_ID,
        last_synced_at=datetime(
            2026,
            8,
            13,
            12,
            0,
            tzinfo=UTC,
        ),
    )
    rows = [
        {
            "user_id": USER_ID,
            "anime_id": 1,
            "rating": 9,
            "status": "completed",
            "updated_at": datetime(
                2026,
                8,
                1,
                12,
                0,
                tzinfo=UTC,
            ),
        },
        {
            "user_id": USER_ID,
            "anime_id": 2,
            "rating": 6,
            "status": "watching",
            "updated_at": datetime(
                2026,
                8,
                2,
                12,
                0,
                tzinfo=UTC,
            ),
        },
    ]
    rates_repository = FakeRatesRepository(rows)

    result = get_recommendations_module.get_recommendations(
        session=object(),
        user_repository=FakeUserRepository(user),
        rates_repository=rates_repository,
        anime_repository=FakeAnimeRepository(),
        user_id=USER_ID,
        bundle=object(),
        artifact_config=object(),
        serving_config=object(),
    )

    interactions = captured["interactions"]

    assert result is expected_result
    assert captured["user_exists"] is True
    assert captured["history_synced"] is True
    assert rates_repository.requested_user_id == USER_ID
    assert interactions["anime_id"].tolist() == [1, 2]
    assert interactions["rating"].tolist() == [9.0, 6.0]


def test_get_recommendations_adds_display_names(monkeypatch):
    user = SimpleNamespace(
        id=USER_ID,
        last_synced_at=datetime(
            2026,
            8,
            13,
            12,
            0,
            tzinfo=UTC,
        ),
    )
    rates_repository = FakeRatesRepository(
        [
            {
                "user_id": USER_ID,
                "anime_id": 10,
                "rating": 9,
                "status": "completed",
                "updated_at": datetime(
                    2026,
                    8,
                    1,
                    12,
                    0,
                    tzinfo=UTC,
                ),
            },
        ]
    )
    anime_repository = FakeAnimeRepository(
        [
            {
                "id": 2,
                "name": "Default Two",
                "russian_name": None,
            },
            {
                "id": 1,
                "name": "Default One",
                "russian_name": "Русское название",
            },
        ]
    )

    monkeypatch.setattr(
        get_recommendations_module,
        "build_recommendations",
        lambda **kwargs: RecommendationResult(
            state=UserState.WARM,
            recommendations=pd.DataFrame(
                {
                    "anime_id": [1, 2],
                    "rank": [1, 2],
                }
            ),
        ),
    )

    result = get_recommendations_module.get_recommendations(
        session=object(),
        user_repository=FakeUserRepository(user),
        anime_repository=anime_repository,
        rates_repository=rates_repository,
        user_id=USER_ID,
        bundle=object(),
        artifact_config=object(),
        serving_config=object(),
    )

    assert anime_repository.requested_anime_ids == [1, 2]
    assert result.recommendations["anime_id"].tolist() == [1, 2]
    assert result.recommendations["display_name"].tolist() == [
        "Русское название",
        "Default Two",
    ]
