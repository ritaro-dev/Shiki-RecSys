import argparse
import logging

from shiki_recsys.application.exceptions import (
    UserNotFoundError,
)
from shiki_recsys.application.sync_user import sync_user
from shiki_recsys.config.settings import get_settings
from shiki_recsys.database.repositories.anime_repository import (
    AnimeRepository,
)
from shiki_recsys.database.repositories.user_rate_svd_repository import (
    UserRateSVDRepository,
)
from shiki_recsys.database.repositories.user_repository import (
    UserRepository,
)
from shiki_recsys.database.session import (
    create_database_engine,
    create_session_factory,
)
from shiki_recsys.integrations.shikimori.client import (
    ShikimoriClient,
)
from shiki_recsys.integrations.shikimori.rate_limiter import (
    ShikimoriRateLimiter,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Синхронизирует историю зарегистрированного пользователя с Shikimori."
        ),
    )

    parser.add_argument(
        "user_id",
        type=int,
        help="Shikimori ID пользователя.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
    )

    args = parse_args()
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

    user_repository = UserRepository()
    anime_repository = AnimeRepository()
    rates_repository = UserRateSVDRepository()

    try:
        logger.info(
            "Начата синхронизация пользователя %s. "
            "Операция может занять длительное время.",
            args.user_id,
        )

        with session_factory() as session:
            user = sync_user(
                session=session,
                client=client,
                user_repository=user_repository,
                anime_repository=anime_repository,
                rates_repository=rates_repository,
                user_id=args.user_id,
            )

        logger.info(
            "Пользователь %s синхронизирован. Время синхронизации: %s.",
            user.id,
            user.last_synced_at,
        )

    except UserNotFoundError as exc:
        logger.error("%s", exc)

    finally:
        client.close()
        engine.dispose()


if __name__ == "__main__":
    main()
