from enum import StrEnum


class UserState(StrEnum):
    """Состояние пользователя для выбора recommendation strategy."""

    USER_NOT_FOUND = "user_not_found"
    NOT_SYNCED = "not_synced"
    EMPTY_HISTORY = "empty_history"
    NO_PREFERENCE_SIGNAL = "no_preference_signal"
    SPARSE_COLD = "sparse_cold"
    PERSONALIZED_COLD = "personalized_cold"
    WARM = "warm"


def classify_user_state(
    *,
    user_exists: bool | None,
    history_synced: bool,
    interaction_count: int,
    supported_positive_count: int,
    supports_personal_retriever: bool,
    min_positive_items: int,
) -> UserState:
    """
    Определяет состояние пользователя для recommendation serving.

    Args:
        user_exists: Подтверждено ли существование пользователя.
        history_synced: Была ли история успешно синхронизирована.
        interaction_count: Число известных взаимодействий.
        supported_positive_count: Число положительных anime,
            поддерживаемых текущим content artifact.
        supports_personal_retriever: Поддерживает ли пользователя personal retriever.
        min_positive_items: Минимум positive items для устойчивого cold-профиля.

    Returns:
        Состояние пользователя.
    """
    if user_exists is False:
        return UserState.USER_NOT_FOUND

    if not history_synced:
        return UserState.NOT_SYNCED

    if interaction_count == 0:
        return UserState.EMPTY_HISTORY

    if supports_personal_retriever:
        return UserState.WARM

    if supported_positive_count == 0:
        return UserState.NO_PREFERENCE_SIGNAL

    if supported_positive_count < min_positive_items:
        return UserState.SPARSE_COLD

    return UserState.PERSONALIZED_COLD
