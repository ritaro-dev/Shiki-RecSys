from collections.abc import Iterator
from typing import Any

from .client import ShikimoriClient
from .exceptions import ShikimoriResponseError
from .queries import ANIMES_CATALOG_QUERY


def iter_anime_pages(
    *,
    client: ShikimoriClient,
    page_size: int,
    max_pages: int | None = None,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """
    Последовательно загружает страницы каталога
    аниме из Shikimori.
    """

    if page_size <= 0:
        raise ValueError("page_size должен быть больше 0.")

    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages должен быть больше 0 или None.")

    page = 1

    while max_pages is None or page <= max_pages:
        result = client.graphql(
            query=ANIMES_CATALOG_QUERY,
            variables={
                "page": page,
                "limit": page_size,
            },
            context_info=(f"anime catalog, page={page}"),
        )

        data = result.get("data")

        if not isinstance(data, dict):
            raise ShikimoriResponseError(
                f"В ответе Shikimori отсутствует объект data на странице {page}."
            )

        anime_list = data.get("animes")

        if not isinstance(anime_list, list):
            raise ShikimoriResponseError(
                f"В ответе Shikimori отсутствует список animes на странице {page}."
            )

        if not anime_list:
            return

        yield page, anime_list

        if len(anime_list) < page_size:
            return

        page += 1
