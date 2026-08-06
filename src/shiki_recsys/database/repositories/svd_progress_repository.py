from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models.svd_collection_progress import (
    SVDCollectionProgress,
)


class SVDCollectionProgressRepository:
    """
    Выполняет операции с состоянием сбора
    пользовательских историй.
    """

    def get_completed_user_ids(
        self,
        session: Session,
    ) -> set[int]:
        """
        Возвращает идентификаторы пользователей,
        сбор которых успешно завершён.
        """

        statement = select(SVDCollectionProgress.user_id).where(
            SVDCollectionProgress.status == "completed"
        )

        return set(session.scalars(statement).all())

    def mark_completed(
        self,
        session: Session,
        *,
        user_id: int,
        rates_saved: int,
    ) -> None:
        """
        Помечает сбор пользователя как завершённый.
        """

        statement = insert(SVDCollectionProgress).values(
            user_id=user_id,
            status="completed",
            rates_saved=rates_saved,
            finished_at=func.now(),
            error_message=None,
        )

        statement = statement.on_conflict_do_update(
            index_elements=[SVDCollectionProgress.user_id],
            set_={
                "status": "completed",
                "rates_saved": rates_saved,
                "finished_at": func.now(),
                "error_message": None,
            },
        )

        session.execute(statement)

    def mark_failed(
        self,
        session: Session,
        *,
        user_id: int,
        error_message: str,
    ) -> None:
        """
        Сохраняет неудачную попытку сбора
        пользовательской истории.
        """

        truncated_message = str(error_message)[:2000]

        statement = insert(SVDCollectionProgress).values(
            user_id=user_id,
            status="failed",
            rates_saved=0,
            finished_at=func.now(),
            error_message=truncated_message,
        )

        statement = statement.on_conflict_do_update(
            index_elements=[SVDCollectionProgress.user_id],
            set_={
                "status": "failed",
                "rates_saved": 0,
                "finished_at": func.now(),
                "error_message": truncated_message,
            },
        )

        session.execute(statement)
