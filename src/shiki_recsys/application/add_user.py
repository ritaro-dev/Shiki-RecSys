from sqlalchemy.orm import Session

from shiki_recsys.database.models.user import User
from shiki_recsys.database.repositories.user_repository import (
    UserRepository,
)

from .exceptions import UserAlreadyExistsError


def add_user(
    *,
    session: Session,
    user_repository: UserRepository,
    user_id: int,
) -> User:
    """
    Регистрирует пользователя Shikimori
    внутри рекомендательной системы.

    История пользователя загружается отдельным
    сценарием sync_user().
    """

    if user_id <= 0:
        raise ValueError("user_id должен быть больше 0.")

    with session.begin():
        existing_user = user_repository.get_by_id(
            session=session,
            user_id=user_id,
        )

        if existing_user is not None:
            raise UserAlreadyExistsError(f"Пользователь {user_id} уже добавлен.")

        user = user_repository.add(
            session=session,
            user_id=user_id,
        )

    return user
