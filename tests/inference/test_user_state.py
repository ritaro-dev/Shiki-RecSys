import pytest

from shiki_recsys.inference.user_state import UserState, classify_user_state


@pytest.mark.parametrize(
    (
        "user_exists",
        "history_synced",
        "interaction_count",
        "supported_positive_count",
        "supports_personal_retriever",
        "min_positive_items",
        "expected",
    ),
    [
        (False, False, 0, 0, False, 5, UserState.USER_NOT_FOUND),
        (True, False, 0, 0, False, 5, UserState.NOT_SYNCED),
        (True, True, 0, 0, True, 5, UserState.EMPTY_HISTORY),
        (True, True, 10, 0, True, 5, UserState.WARM),
        (True, True, 10, 0, False, 5, UserState.NO_PREFERENCE_SIGNAL),
        (True, True, 10, 2, False, 5, UserState.SPARSE_COLD),
        (True, True, 10, 5, False, 5, UserState.PERSONALIZED_COLD),
        (True, True, 10, 8, False, 5, UserState.PERSONALIZED_COLD),
    ],
)
def test_classify_user_state(
    user_exists: bool,
    history_synced: bool,
    interaction_count: int,
    supported_positive_count: int,
    supports_personal_retriever: bool,
    min_positive_items: int,
    expected: UserState,
) -> None:
    """Проверяет классификацию состояния пользователя."""
    result = classify_user_state(
        user_exists=user_exists,
        history_synced=history_synced,
        interaction_count=interaction_count,
        supported_positive_count=supported_positive_count,
        supports_personal_retriever=supports_personal_retriever,
        min_positive_items=min_positive_items,
    )

    assert result == expected
