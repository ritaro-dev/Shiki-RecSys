class ApplicationError(Exception):
    """
    Базовая ошибка сценариев приложения.
    """


class UserAlreadyExistsError(ApplicationError):
    """
    Пользователь уже добавлен в систему.
    """


class UserNotFoundError(ApplicationError):
    """
    Пользователь не найден в системе.
    """
