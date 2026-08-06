from typing import Any

from .client import ShikimoriClient
from .exceptions import ShikimoriResponseError
from .queries import USER_HISTORY_QUERY

USER_HISTORY_PAGE_SIZE = 50


def fetch_user_history(
    *,
    client: ShikimoriClient,
    user_id: int,
) -> list[dict[str, Any]]:
    """
    Загружает полную историю аниме пользователя
    из Shikimori.
    """

    if user_id <= 0:
        raise ValueError("user_id должен быть больше 0.")

    page = 1
    history: list[dict[str, Any]] = []

    while True:
        result = client.graphql(
            query=USER_HISTORY_QUERY,
            variables={
                "userId": str(user_id),
                "page": page,
                "limit": USER_HISTORY_PAGE_SIZE,
            },
            context_info=(f"user_id={user_id}, page={page}"),
        )

        data = result.get("data")

        if not isinstance(data, dict):
            raise ShikimoriResponseError(
                "В ответе Shikimori отсутствует объект data "
                f"для user_id={user_id}, page={page}."
            )

        rates = data.get("userRates")

        if not isinstance(rates, list):
            raise ShikimoriResponseError(
                "В ответе Shikimori отсутствует список userRates "
                f"для user_id={user_id}, page={page}."
            )

        history.extend(rates)

        if len(rates) < USER_HISTORY_PAGE_SIZE:
            break

        page += 1

    return history
