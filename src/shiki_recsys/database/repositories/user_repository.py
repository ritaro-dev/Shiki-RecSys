from datetime import datetime

from sqlalchemy.orm import Session

from ..models.user import User


class UserRepository:
    """
    Выполняет операции с пользователями
    рекомендательной системы.
    """

    def get_by_id(
        self,
        session: Session,
        *,
        user_id: int,
    ) -> User | None:
        """
        Возвращает пользователя по его Shikimori ID.
        """

        return session.get(User, user_id)

    def add(
        self,
        session: Session,
        *,
        user_id: int,
    ) -> User:
        """
        Добавляет пользователя в текущую сессию.

        Не выполняет commit: транзакцией управляет
        вызывающий код.
        """

        user = User(id=user_id)
        session.add(user)

        return user

    def mark_synced(
        self,
        user: User,
        *,
        synced_at: datetime,
    ) -> None:
        """
        Записывает время последней успешной
        синхронизации пользователя.
        """

        if synced_at.tzinfo is None:
            raise ValueError("synced_at должен содержать часовой пояс.")

        user.last_synced_at = synced_at
