import logging

from shiki_recsys.config.settings import get_settings
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
from shiki_recsys.database.session import (
    create_database_engine,
    create_session_factory,
)
from shiki_recsys.ingestion.user_history_collector import (
    collect_seed_users,
)
from shiki_recsys.integrations.shikimori.client import (
    ShikimoriClient,
)
from shiki_recsys.integrations.shikimori.rate_limiter import (
    ShikimoriRateLimiter,
)


def main() -> None:
    """
    Собирает полные истории seed-пользователей
    и сохраняет их в user_rates_svd.
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

    anime_repository = AnimeRepository()
    seed_user_repository = LegacySeedUserRepository()
    rates_repository = UserRateSVDRepository()
    progress_repository = SVDCollectionProgressRepository()

    try:
        with session_factory() as session:
            collect_seed_users(
                session=session,
                client=client,
                anime_repository=anime_repository,
                seed_user_repository=(seed_user_repository),
                rates_repository=rates_repository,
                progress_repository=(progress_repository),
                total_users=1000,
            )

    finally:
        client.close()
        engine.dispose()


if __name__ == "__main__":
    main()
