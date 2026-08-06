from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from shiki_recsys.config.settings import Settings


def create_database_engine(
    settings: Settings,
) -> Engine:
    """
    Создаёт SQLAlchemy Engine для PostgreSQL.
    """

    database_url = URL.create(
        drivername="postgresql+psycopg2",
        username=settings.db_user,
        password=(settings.db_password.get_secret_value()),
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_database,
    )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


def create_session_factory(
    engine: Engine,
) -> sessionmaker[Session]:
    """
    Создаёт фабрику SQLAlchemy-сессий.
    """

    return sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
