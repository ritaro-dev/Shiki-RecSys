from typing import Annotated

from fastapi import APIRouter, Depends, Path, status
from sqlalchemy.orm import Session

from shiki_recsys.api.dependencies import (
    get_anime_repository,
    get_rates_repository,
    get_session,
    get_shikimori_client,
    get_user_repository,
)
from shiki_recsys.api.schemas.users import (
    CreateUserRequest,
    UserResponse,
)
from shiki_recsys.application.add_user import add_user
from shiki_recsys.application.sync_user import sync_user
from shiki_recsys.database.repositories.anime_repository import (
    AnimeRepository,
)
from shiki_recsys.database.repositories.user_rate_svd_repository import (
    UserRateSVDRepository,
)
from shiki_recsys.database.repositories.user_repository import (
    UserRepository,
)
from shiki_recsys.integrations.shikimori.client import (
    ShikimoriClient,
)

router = APIRouter(
    prefix="/users",
    tags=["users"],
)


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    payload: CreateUserRequest,
    session: Annotated[
        Session,
        Depends(get_session),
    ],
    user_repository: Annotated[
        UserRepository,
        Depends(get_user_repository),
    ],
) -> UserResponse:
    """
    Регистрирует пользователя по его Shikimori ID.

    История пользователя этим запросом
    не синхронизируется.
    """

    user = add_user(
        session=session,
        user_repository=user_repository,
        user_id=payload.user_id,
    )

    return UserResponse.model_validate(user)


@router.post(
    "/{user_id}/sync",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
)
def synchronize_user(
    user_id: Annotated[
        int,
        Path(
            gt=0,
            description="Shikimori ID пользователя.",
        ),
    ],
    session: Annotated[
        Session,
        Depends(get_session),
    ],
    client: Annotated[
        ShikimoriClient,
        Depends(get_shikimori_client),
    ],
    user_repository: Annotated[
        UserRepository,
        Depends(get_user_repository),
    ],
    anime_repository: Annotated[
        AnimeRepository,
        Depends(get_anime_repository),
    ],
    rates_repository: Annotated[
        UserRateSVDRepository,
        Depends(get_rates_repository),
    ],
) -> UserResponse:
    """
    Синхронизирует историю зарегистрированного
    пользователя с Shikimori.

    Запрос выполняется синхронно и может занять
    длительное время.
    """

    user = sync_user(
        session=session,
        client=client,
        user_repository=user_repository,
        anime_repository=anime_repository,
        rates_repository=rates_repository,
        user_id=user_id,
    )

    return UserResponse.model_validate(user)
