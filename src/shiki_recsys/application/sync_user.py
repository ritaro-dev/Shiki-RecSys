from datetime import UTC, datetime

from sqlalchemy.orm import Session

from shiki_recsys.database.models.user import User
from shiki_recsys.database.repositories.anime_repository import (
    AnimeRepository,
)
from shiki_recsys.database.repositories.user_rate_svd_repository import (
    UserRateSVDRepository,
)
from shiki_recsys.database.repositories.user_repository import (
    UserRepository,
)
from shiki_recsys.ingestion.normalization import (
    normalize_user_history,
)
from shiki_recsys.integrations.shikimori.client import (
    ShikimoriClient,
)
from shiki_recsys.integrations.shikimori.user_history import (
    fetch_user_history,
)

from .exceptions import UserNotFoundError


def sync_user(
    *,
    session: Session,
    client: ShikimoriClient,
    user_repository: UserRepository,
    anime_repository: AnimeRepository,
    rates_repository: UserRateSVDRepository,
    user_id: int,
) -> User:
    """
    Загружает актуальную историю пользователя
    из Shikimori и заменяет сохранённую историю.
    """

    if user_id <= 0:
        raise ValueError("user_id должен быть больше 0.")

    # Выполняем короткую проверку до длительного
    # обращения к Shikimori.
    with session.begin():
        user = user_repository.get_by_id(
            session=session,
            user_id=user_id,
        )

        if user is None:
            raise UserNotFoundError(f"Пользователь {user_id} не найден.")

        allowed_anime_ids = anime_repository.get_all_ids(
            session=session,
        )

    # Во время медленной загрузки истории
    # транзакция PostgreSQL не остаётся открытой.
    history = fetch_user_history(
        client=client,
        user_id=user_id,
    )

    rate_rows = normalize_user_history(
        user_id=user_id,
        history=history,
        allowed_anime_ids=allowed_anime_ids,
    )

    synced_at = datetime.now(UTC)

    # Замена истории и обновление времени
    # синхронизации выполняются атомарно.
    with session.begin():
        user = user_repository.get_by_id(
            session=session,
            user_id=user_id,
        )

        if user is None:
            raise UserNotFoundError(f"Пользователь {user_id} не найден.")

        rates_repository.delete_by_user_id(
            session=session,
            user_id=user_id,
        )

        rates_repository.upsert_many(
            session=session,
            rate_rows=rate_rows,
        )

        user_repository.mark_synced(
            user,
            synced_at=synced_at,
        )

    return user
