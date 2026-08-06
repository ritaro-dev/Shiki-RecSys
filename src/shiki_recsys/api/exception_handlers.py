from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from shiki_recsys.application.exceptions import (
    UserAlreadyExistsError,
    UserNotFoundError,
)
from shiki_recsys.integrations.shikimori.exceptions import (
    ShikimoriError,
)


async def handle_user_already_exists(
    request: Request,
    exc: UserAlreadyExistsError,
) -> JSONResponse:
    """
    Преобразует ошибку повторной регистрации
    в HTTP 409 Conflict.
    """

    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "detail": str(exc),
        },
    )


async def handle_user_not_found(
    request: Request,
    exc: UserNotFoundError,
) -> JSONResponse:
    """
    Преобразует отсутствие пользователя
    в HTTP 404 Not Found.
    """

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "detail": str(exc),
        },
    )


async def handle_shikimori_error(
    request: Request,
    exc: ShikimoriError,
) -> JSONResponse:
    """
    Преобразует ошибку внешнего API Shikimori
    в HTTP 502 Bad Gateway.
    """

    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "detail": ("Не удалось получить данные из Shikimori."),
        },
    )


def register_exception_handlers(
    app: FastAPI,
) -> None:
    """
    Регистрирует обработчики ошибок
    в FastAPI-приложении.
    """

    app.add_exception_handler(
        UserAlreadyExistsError,
        handle_user_already_exists,
    )

    app.add_exception_handler(
        UserNotFoundError,
        handle_user_not_found,
    )

    app.add_exception_handler(
        ShikimoriError,
        handle_shikimori_error,
    )
