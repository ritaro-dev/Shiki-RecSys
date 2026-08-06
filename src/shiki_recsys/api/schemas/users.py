from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CreateUserRequest(BaseModel):
    """
    Тело запроса на регистрацию пользователя.
    """

    user_id: int = Field(
        gt=0,
        description="Shikimori ID пользователя.",
    )


class UserResponse(BaseModel):
    """
    Данные зарегистрированного пользователя.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: int
    created_at: datetime
    last_synced_at: datetime | None
