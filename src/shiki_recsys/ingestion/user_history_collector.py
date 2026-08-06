import logging

from sqlalchemy.orm import Session

from shiki_recsys.database.repositories.anime_repository import (
    AnimeRepository,
)
from shiki_recsys.database.repositories.legacy.legacy_seed_user_repository import (
    LegacySeedUserRepository,
)
from shiki_recsys.database.repositories.svd_progress_repository import (
    SVDCollectionProgressRepository,
)
from shiki_recsys.database.repositories.user_rate_svd_repository import (
    UserRateSVDRepository,
)
from shiki_recsys.integrations.shikimori.client import (
    ShikimoriClient,
)
from shiki_recsys.integrations.shikimori.user_history import (
    fetch_user_history,
)

from .normalization import normalize_user_history

logger = logging.getLogger(__name__)


def collect_one_user(
    *,
    session: Session,
    client: ShikimoriClient,
    rates_repository: UserRateSVDRepository,
    progress_repository: SVDCollectionProgressRepository,
    user_id: int,
    allowed_anime_ids: set[int],
) -> int:
    """
    Загружает и сохраняет историю одного пользователя.

    Взаимодействия и статус completed сохраняются
    в одной транзакции.
    """

    history = fetch_user_history(
        client=client,
        user_id=user_id,
    )

    rate_rows = normalize_user_history(
        user_id=user_id,
        history=history,
        allowed_anime_ids=allowed_anime_ids,
    )

    with session.begin():
        saved_count = rates_repository.upsert_many(
            session=session,
            rate_rows=rate_rows,
        )

        progress_repository.mark_completed(
            session=session,
            user_id=user_id,
            rates_saved=saved_count,
        )

    logger.info(
        "Пользователь %s обработан: получено %s, сохранено %s.",
        user_id,
        len(history),
        saved_count,
    )

    return saved_count


def collect_seed_users(
    *,
    session: Session,
    client: ShikimoriClient,
    anime_repository: AnimeRepository,
    seed_user_repository: LegacySeedUserRepository,
    rates_repository: UserRateSVDRepository,
    progress_repository: SVDCollectionProgressRepository,
    total_users: int,
) -> None:
    """
    Собирает истории стратифицированной выборки
    пользователей из устаревшей таблицы user_rates.

    Ранее успешно обработанные пользователи
    пропускаются.
    """

    if total_users <= 0:
        raise ValueError("total_users должен быть больше 0.")

    # Эти запросы выполняются в короткой отдельной
    # транзакции. Во время сетевых запросов к Shikimori
    # транзакция PostgreSQL не остаётся открытой.
    with session.begin():
        seed_user_ids = seed_user_repository.get_stratified_user_ids(
            session=session,
            total_users=total_users,
        )

        completed_user_ids = progress_repository.get_completed_user_ids(
            session=session,
        )

        allowed_anime_ids = anime_repository.get_all_ids(
            session=session,
        )

    logger.info(
        "Найдено seed-пользователей: %s. Ранее завершено: %s.",
        len(seed_user_ids),
        len(completed_user_ids.intersection(seed_user_ids)),
    )

    for index, user_id in enumerate(
        seed_user_ids,
        start=1,
    ):
        if user_id in completed_user_ids:
            logger.info(
                "[%s/%s] Пользователь %s пропущен: уже обработан.",
                index,
                len(seed_user_ids),
                user_id,
            )
            continue

        logger.info(
            "[%s/%s] Начата обработка пользователя %s.",
            index,
            len(seed_user_ids),
            user_id,
        )

        try:
            saved_count = collect_one_user(
                session=session,
                client=client,
                rates_repository=rates_repository,
                progress_repository=progress_repository,
                user_id=user_id,
                allowed_anime_ids=allowed_anime_ids,
            )

            completed_user_ids.add(user_id)

            logger.info(
                "[%s/%s] Пользователь %s завершён. Сохранено взаимодействий: %s.",
                index,
                len(seed_user_ids),
                user_id,
                saved_count,
            )

        except Exception as exc:  # noqa: BLE001
            # На случай ошибки вне блока session.begin().
            if session.in_transaction():
                session.rollback()

            error_message = f"{type(exc).__name__}: {exc}"

            try:
                # Ошибка записывается отдельной транзакцией,
                # поскольку основная транзакция пользователя
                # уже отменена.
                with session.begin():
                    progress_repository.mark_failed(
                        session=session,
                        user_id=user_id,
                        error_message=error_message,
                    )

            except Exception:
                logger.exception(
                    "Не удалось записать статус failed для пользователя %s.",
                    user_id,
                )

            logger.error(
                "[%s/%s] Ошибка обработки пользователя %s: %s",
                index,
                len(seed_user_ids),
                user_id,
                error_message,
            )
