import logging
import time
from typing import Any

import requests
from requests.exceptions import RequestException

from .exceptions import (
    ShikimoriError,
    ShikimoriGraphQLError,
    ShikimoriHTTPError,
    ShikimoriNetworkError,
    ShikimoriResponseError,
)
from .rate_limiter import ShikimoriRateLimiter

logger = logging.getLogger(__name__)


class ShikimoriClient:
    """
    Клиент для выполнения GraphQL-запросов
    к Shikimori API.
    """

    def __init__(
        self,
        *,
        graphql_url: str,
        user_agent: str,
        limiter: ShikimoriRateLimiter,
        timeout_seconds: float,
        max_retries: int,
        session: requests.Session | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds должен быть больше 0.")

        if max_retries <= 0:
            raise ValueError("max_retries должен быть больше 0.")

        self._graphql_url = graphql_url
        self._limiter = limiter
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

        self._session = session or requests.Session()

        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )

    def close(self) -> None:
        """
        Закрывает HTTP-сессию клиента.
        """

        self._session.close()

    def graphql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        context_info: str = "",
    ) -> dict[str, Any]:
        """
        Выполняет GraphQL-запрос с ограничением частоты
        и повторными попытками при временных ошибках.
        """

        if not query.strip():
            raise ValueError("GraphQL-запрос не должен быть пустым.")

        payload = {
            "query": query,
            "variables": variables or {},
        }

        for attempt in range(
            1,
            self._max_retries + 1,
        ):
            self._limiter.wait()

            try:
                response = self._session.post(
                    self._graphql_url,
                    json=payload,
                    timeout=self._timeout_seconds,
                )

            except RequestException as exc:
                if attempt == self._max_retries:
                    raise ShikimoriNetworkError(
                        "Не удалось выполнить запрос "
                        f"после {self._max_retries} попыток. "
                        f"{context_info}: {exc}"
                    ) from exc

                wait_time = attempt * 3

                logger.warning(
                    "Сетевая ошибка. %s Повтор через %s сек.",
                    context_info,
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            if response.status_code == 200:
                try:
                    result = response.json()

                except ValueError as exc:
                    raise ShikimoriResponseError(
                        f"Shikimori вернул некорректный JSON. {context_info}"
                    ) from exc

                if not isinstance(result, dict):
                    raise ShikimoriResponseError(
                        f"Shikimori вернул неожиданный формат ответа. {context_info}"
                    )

                if result.get("errors"):
                    raise ShikimoriGraphQLError(
                        "Shikimori вернул GraphQL-ошибку. "
                        f"{context_info}: "
                        f"{result['errors']}"
                    )

                return result

            if response.status_code == 429:
                if attempt == self._max_retries:
                    raise ShikimoriHTTPError(
                        "Shikimori продолжает возвращать "
                        f"HTTP 429 после "
                        f"{self._max_retries} попыток. "
                        f"{context_info}"
                    )

                retry_after = response.headers.get("Retry-After")

                try:
                    wait_time = float(retry_after)

                    if wait_time <= 0:
                        raise ValueError

                except (TypeError, ValueError):
                    wait_time = attempt * 10

                logger.warning(
                    "Получен HTTP 429. %s Повтор через %.1f сек.",
                    context_info,
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            if response.status_code >= 500:
                if attempt == self._max_retries:
                    raise ShikimoriHTTPError(
                        "Shikimori продолжает возвращать "
                        f"HTTP {response.status_code} после "
                        f"{self._max_retries} попыток. "
                        f"{context_info}"
                    )

                wait_time = attempt * 3

                logger.warning(
                    "Получен HTTP %s. %s Повтор через %s сек.",
                    response.status_code,
                    context_info,
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            raise ShikimoriHTTPError(
                f"Shikimori вернул HTTP "
                f"{response.status_code}. "
                f"{context_info}. "
                f"Ответ: {response.text[:500]}"
            )

        raise ShikimoriError(f"Не удалось выполнить запрос. {context_info}")
