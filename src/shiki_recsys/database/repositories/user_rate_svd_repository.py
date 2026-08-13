from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models.user_rate_svd import UserRateSVD


class UserRateSVDRepository:
    """
    Выполняет операции с таблицей user_rates_svd.
    """

    def upsert_many(
        self,
        session: Session,
        rate_rows: Sequence[dict[str, Any]],
    ) -> int:
        """
        Добавляет или обновляет пачку взаимодействий.

        Не выполняет commit: транзакцией управляет
        вызывающий код.
        """

        if not rate_rows:
            return 0

        statement = insert(UserRateSVD)

        statement = statement.on_conflict_do_update(
            constraint="unique_svd_user_anime",
            set_={
                "rating": statement.excluded.rating,
                "status": statement.excluded.status,
                "updated_at": (statement.excluded.updated_at),
            },
        )

        session.execute(
            statement,
            list(rate_rows),
        )

        return len(rate_rows)

    def delete_by_user_id(
        self,
        session: Session,
        *,
        user_id: int,
    ) -> None:
        """
        Удаляет все сохранённые взаимодействия
        указанного пользователя.

        Не выполняет commit: транзакцией управляет
        вызывающий сценарий.
        """

        statement = delete(UserRateSVD).where(UserRateSVD.user_id == user_id)

        session.execute(statement)

    def get_by_user_id(
        self,
        session: Session,
        *,
        user_id: int,
    ) -> list[dict[str, object]]:
        """
        Return interactions stored for a user.

        Args:
            session: Database session.
            user_id: Shikimori user ID.

        Returns:
            Stored user interactions.
        """
        statement = (
            select(
                UserRateSVD.user_id,
                UserRateSVD.anime_id,
                UserRateSVD.rating,
                UserRateSVD.status,
                UserRateSVD.updated_at,
            )
            .where(UserRateSVD.user_id == user_id)
            .order_by(
                UserRateSVD.updated_at,
                UserRateSVD.anime_id,
            )
        )

        rows = session.execute(statement).mappings().all()

        return [dict(row) for row in rows]

    def get_all(
        self,
        session: Session,
    ) -> list[dict[str, object]]:
        """
        Возвращает все сохранённые взаимодействия.

        Загружает результат целиком в память.
        Не изменяет данные и не выполняет commit.
        """

        statement = select(
            UserRateSVD.user_id,
            UserRateSVD.anime_id,
            UserRateSVD.rating,
            UserRateSVD.status,
            UserRateSVD.updated_at,
        ).order_by(
            UserRateSVD.user_id,
            UserRateSVD.updated_at,
            UserRateSVD.anime_id,
        )

        rows = session.execute(statement).mappings().all()

        return [dict(row) for row in rows]
