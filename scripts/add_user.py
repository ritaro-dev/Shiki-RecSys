import argparse
import logging

from shiki_recsys.application.add_user import add_user
from shiki_recsys.application.exceptions import (
    UserAlreadyExistsError,
)
from shiki_recsys.config.settings import get_settings
from shiki_recsys.database.repositories.user_repository import (
    UserRepository,
)
from shiki_recsys.database.session import (
    create_database_engine,
    create_session_factory,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Регистрирует пользователя Shikimori в рекомендательной системе."),
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

    user_repository = UserRepository()

    try:
        with session_factory() as session:
            user = add_user(
                session=session,
                user_repository=user_repository,
                user_id=args.user_id,
            )

        logger.info(
            "Пользователь %s зарегистрирован. История ещё не синхронизирована.",
            user.id,
        )

    except UserAlreadyExistsError as exc:
        logger.error("%s", exc)

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
