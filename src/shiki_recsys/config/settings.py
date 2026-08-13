from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """
    Настройки проекта.

    Значения можно переопределить через переменные
    окружения или локальный файл .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL
    db_host: str = "localhost"
    db_port: int = 5432
    db_database: str = "shiki_db"
    db_user: str = "shiki_app"
    db_password: SecretStr

    # Shikimori GraphQL API
    shikimori_graphql_url: str = "https://shikimori.io/api/graphql"

    shikimori_user_agent: str = "ShikiRecsys/1.0 (Personal Study Project)"

    shikimori_max_retries: int = 5
    shikimori_timeout_seconds: float = 15.0

    shikimori_min_interval_seconds: float = 0.8

    # Model artifacts
    artifacts_dir: Path = Path("artifacts")

    # Recommendation serving
    recommendation_top_k: int = 20
    recommendation_min_positive_items: int

    @property
    def database_connection_kwargs(
        self,
    ) -> dict[str, object]:
        """
        Возвращает настройки подключения в формате,
        который принимает psycopg2.connect.
        """

        return {
            "host": self.db_host,
            "port": self.db_port,
            "database": self.db_database,
            "user": self.db_user,
            "password": (self.db_password.get_secret_value()),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Возвращает один общий объект настроек.
    """

    return Settings()
