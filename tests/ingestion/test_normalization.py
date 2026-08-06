from datetime import UTC, datetime
from decimal import Decimal

from shiki_recsys.ingestion.normalization import (
    normalize_anime,
    normalize_user_history,
)


def test_normalize_anime() -> None:
    raw_anime = {
        "id": "1",
        "name": "Test Anime",
        "russian": "",
        "kind": "tv",
        "status": "released",
        "score": "8.25",
        "episodes": 12,
        "duration": 24,
        "rating": "pg_13",
        "genres": [
            {"name": "Drama"},
            {"name": "Fantasy"},
            {"name": ""},
        ],
        "studios": [
            {"name": "Test Studio"},
        ],
        "statusesStats": [
            {
                "status": "completed",
                "count": 100,
            },
            {
                "status": "watching",
                "count": 20,
            },
            {
                "status": "on_hold",
                "count": 5,
            },
        ],
    }

    result = normalize_anime(raw_anime)

    assert result["id"] == 1
    assert result["name"] == "Test Anime"

    # Пустое русское название заменяется
    # оригинальным названием.
    assert result["russian_name"] == "Test Anime"

    assert result["score"] == Decimal("8.25")
    assert result["genres"] == ["Drama", "Fantasy"]
    assert result["studios"] == ["Test Studio"]

    # watching и on_hold объединяются.
    assert result["stat_watching"] == 25
    assert result["stat_completed"] == 100


def test_normalize_user_history() -> None:
    history = [
        {
            "score": 8,
            "status": "completed",
            "updatedAt": "2026-08-02T15:30:00+03:00",
            "anime": {
                "id": "1",
            },
        },
        {
            # Нулевая оценка должна сохраняться.
            "score": 0,
            "status": "planned",
            "updatedAt": "2026-08-02T12:00:00Z",
            "anime": {
                "id": "2",
            },
        },
        {
            # Аниме отсутствует в разрешённом каталоге.
            "score": 7,
            "status": "completed",
            "updatedAt": "2026-08-02T12:00:00Z",
            "anime": {
                "id": "999",
            },
        },
        {
            # Некорректная оценка.
            "score": 15,
            "status": "completed",
            "updatedAt": "2026-08-02T12:00:00Z",
            "anime": {
                "id": "1",
            },
        },
    ]

    result = normalize_user_history(
        user_id=100,
        history=history,
        allowed_anime_ids={1, 2},
    )

    assert len(result) == 2

    assert result[0] == {
        "user_id": 100,
        "anime_id": 1,
        "rating": 8,
        "status": "completed",
        "updated_at": datetime(
            2026,
            8,
            2,
            12,
            30,
            tzinfo=UTC,
        ),
    }

    assert result[1]["anime_id"] == 2
    assert result[1]["rating"] == 0
    assert result[1]["updated_at"] == datetime(
        2026,
        8,
        2,
        12,
        0,
        tzinfo=UTC,
    )
