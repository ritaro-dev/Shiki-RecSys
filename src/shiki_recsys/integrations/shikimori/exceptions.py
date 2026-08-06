class ShikimoriError(RuntimeError):
    """
    Базовая ошибка при работе с Shikimori API.
    """


class ShikimoriNetworkError(ShikimoriError):
    """
    Ошибка сетевого соединения с Shikimori.
    """


class ShikimoriHTTPError(ShikimoriError):
    """
    Ошибка HTTP-ответа Shikimori.
    """


class ShikimoriGraphQLError(ShikimoriError):
    """
    Ошибка, возвращённая GraphQL API.
    """


class ShikimoriResponseError(ShikimoriError):
    """
    Некорректный или неожиданный ответ Shikimori.
    """
