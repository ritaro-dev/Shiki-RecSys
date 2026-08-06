import logging

from sqlalchemy.orm import Session

from shiki_recsys.database.repositories.anime_repository import (
    AnimeRepository,
)
from shiki_recsys.integrations.shikimori.anime_catalog import (
    iter_anime_pages,
)
from shiki_recsys.integrations.shikimori.client import (
    ShikimoriClient,
)

from .normalization import normalize_animes

logger = logging.getLogger(__name__)


def collect_all_animes(
    *,
    session: Session,
    client: ShikimoriClient,
    repository: AnimeRepository,
    per_page: int,
    max_pages: int | None = None,
) -> int:
    """
    Загружает каталог аниме из Shikimori
    и сохраняет его в PostgreSQL.
    """

    total_saved = 0

    logger.info("Начат сбор каталога аниме.")

    for page, anime_list in iter_anime_pages(
        client=client,
        page_size=per_page,
        max_pages=max_pages,
    ):
        anime_rows = normalize_animes(anime_list)

        # Каждая страница сохраняется отдельной
        # транзакцией.
        with session.begin():
            saved_count = repository.upsert_many(
                session=session,
                anime_rows=anime_rows,
            )

        total_saved += saved_count

        logger.info(
            "Обработана страница %s: получено %s, всего сохранено %s.",
            page,
            len(anime_list),
            total_saved,
        )

    logger.info(
        "Сбор каталога завершён. Всего обработано: %s.",
        total_saved,
    )

    return total_saved
