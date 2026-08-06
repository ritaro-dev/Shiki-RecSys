import logging

from shiki_recsys.config.settings import get_settings
from shiki_recsys.database.repositories.anime_repository import (
    AnimeRepository,
)
from shiki_recsys.database.session import (
    create_database_engine,
    create_session_factory,
)
from shiki_recsys.ingestion.anime_collector import (
    collect_all_animes,
)
from shiki_recsys.integrations.shikimori.client import (
    ShikimoriClient,
)
from shiki_recsys.integrations.shikimori.rate_limiter import (
    ShikimoriRateLimiter,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Собирает каталог аниме из Shikimori
    и сохраняет его в PostgreSQL.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
    )

    settings = get_settings()

    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    limiter = ShikimoriRateLimiter(
        min_interval_seconds=(settings.shikimori_min_interval_seconds),
    )

    client = ShikimoriClient(
        graphql_url=settings.shikimori_graphql_url,
        user_agent=settings.shikimori_user_agent,
        limiter=limiter,
        timeout_seconds=(settings.shikimori_timeout_seconds),
        max_retries=settings.shikimori_max_retries,
    )

    repository = AnimeRepository()

    try:
        with session_factory() as session:
            total_saved = collect_all_animes(
                session=session,
                client=client,
                repository=repository,
                per_page=50,
                max_pages=None,
            )

        logger.info(
            "Сбор завершён. Обработано аниме: %s.",
            total_saved,
        )

    finally:
        client.close()
        engine.dispose()


if __name__ == "__main__":
    main()
