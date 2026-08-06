from logging.config import fileConfig

from alembic import context
from shiki_recsys.config.settings import get_settings
from shiki_recsys.database.base import Base
from shiki_recsys.database.session import (
    create_database_engine,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


target_metadata = Base.metadata

LEGACY_TABLE_NAMES = {
    "user_rates",
}


def include_object(
    object_: object,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: object | None,
) -> bool:
    """
    Исключает legacy-таблицы из автоматического
    сравнения схемы Alembic.
    """

    return not (type_ == "table" and name in LEGACY_TABLE_NAMES)


def run_migrations_offline() -> None:
    """
    Формирует SQL миграций без подключения
    к PostgreSQL.
    """

    settings = get_settings()
    engine = create_database_engine(settings)

    try:
        database_url = engine.url.render_as_string(hide_password=False)

        context.configure(
            url=database_url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={
                "paramstyle": "named",
            },
            compare_type=True,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()

    finally:
        engine.dispose()


def run_migrations_online() -> None:
    """
    Выполняет миграции с подключением
    к PostgreSQL.
    """

    settings = get_settings()
    engine = create_database_engine(settings)

    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                include_object=include_object,
            )

            with context.begin_transaction():
                context.run_migrations()

    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
