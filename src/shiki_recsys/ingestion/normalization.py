from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any


def _parse_score(value: Any) -> Decimal | None:
    """
    Преобразует оценку Shikimori в Decimal.
    """

    if value in (None, ""):
        return None

    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def normalize_anime(
    anime: dict[str, Any],
) -> dict[str, Any]:
    """
    Преобразует данные одного аниме из формата
    Shikimori во внутренний формат проекта.
    """

    genres = [genre["name"] for genre in anime.get("genres", []) if genre.get("name")]

    studios = [
        studio["name"] for studio in anime.get("studios", []) if studio.get("name")
    ]

    stats = {
        item["status"]: item["count"]
        for item in anime.get("statusesStats", [])
        if item.get("status") is not None
    }

    return {
        "id": int(anime["id"]),
        "name": anime["name"],
        "russian_name": (anime.get("russian") or anime["name"]),
        "kind": anime.get("kind"),
        "status": anime.get("status"),
        "score": _parse_score(anime.get("score")),
        "episodes": anime.get("episodes"),
        "duration": anime.get("duration"),
        "rating": anime.get("rating"),
        "genres": genres,
        "studios": studios,
        "stat_completed": stats.get("completed", 0),
        "stat_dropped": stats.get("dropped", 0),
        "stat_watching": (stats.get("watching", 0) + stats.get("on_hold", 0)),
        "stat_planned": stats.get("planned", 0),
    }


def normalize_animes(
    animes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Преобразует список аниме из формата
    Shikimori во внутренний формат проекта.
    """

    return [normalize_anime(anime) for anime in animes]


def _parse_updated_at(
    value: Any,
) -> datetime | None:
    """
    Преобразует дату Shikimori в datetime
    и приводит её к часовому поясу UTC.
    """

    if value in (None, ""):
        return None

    if isinstance(value, datetime):
        parsed_datetime = value

    elif isinstance(value, str):
        normalized_value = value.strip()

        if normalized_value.endswith("Z"):
            normalized_value = normalized_value[:-1] + "+00:00"

        try:
            parsed_datetime = datetime.fromisoformat(normalized_value)
        except ValueError:
            return None

    else:
        return None

    if parsed_datetime.tzinfo is None:
        return None

    return parsed_datetime.astimezone(UTC)


def normalize_user_history(
    *,
    user_id: int,
    history: list[dict[str, Any]],
    allowed_anime_ids: set[int],
) -> list[dict[str, Any]]:
    """
    Преобразует историю пользователя Shikimori
    в строки для таблицы user_rates_svd.

    Сохраняются оценки от 0 до 10 и только аниме,
    присутствующие в allowed_anime_ids.
    """

    normalized_rates: list[dict[str, Any]] = []

    for rate in history:
        anime = rate.get("anime")

        if not isinstance(anime, dict):
            continue

        raw_anime_id = anime.get("id")

        if raw_anime_id is None:
            continue

        try:
            anime_id = int(raw_anime_id)
            rating = int(rate.get("score", 0))
        except (TypeError, ValueError):
            continue

        if rating < 0 or rating > 10:
            continue

        if anime_id not in allowed_anime_ids:
            continue

        normalized_rates.append(
            {
                "user_id": int(user_id),
                "anime_id": anime_id,
                "rating": rating,
                "status": rate.get("status"),
                "updated_at": _parse_updated_at(rate.get("updatedAt")),
            }
        )

    return normalized_rates
