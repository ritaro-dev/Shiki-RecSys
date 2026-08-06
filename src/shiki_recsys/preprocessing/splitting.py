from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InteractionSplit:
    """Хранит результаты хронологического разделения взаимодействий."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    evaluation_user_ids: frozenset[int]
    train_only_user_ids: frozenset[int]


def chronological_split(
    interactions: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    min_interactions_per_user: int,
) -> InteractionSplit:
    """
    Выполняет хронологическое разделение взаимодействий пользователей.

    Args:
        interactions: Подготовленный набор взаимодействий.
        validation_fraction: Доля взаимодействий для валидации.
        test_fraction: Доля взаимодействий для тестирования.
        min_interactions_per_user: Минимальное количество взаимодействий
            пользователя для участия в оценке.

    Returns:
        Обучающий, валидационный и тестовый наборы, идентификаторы
        оцениваемых пользователей и пользователей, представленных
        только в обучающем наборе.

    Raises:
        ValueError: Если параметры разделения некорректны или
            отсутствуют необходимые столбцы.
    """

    fractions = {
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    }

    invalid_fractions = [name for name, value in fractions.items() if not 0 < value < 1]

    if invalid_fractions:
        raise ValueError(
            f"Каждая доля должна находиться в диапазоне от 0 до 1: {invalid_fractions}."
        )

    if validation_fraction + test_fraction >= 1:
        raise ValueError(
            "Сумма validation_fraction и test_fraction должна быть меньше 1."
        )

    if min_interactions_per_user <= 0:
        raise ValueError("min_interactions_per_user должен быть больше 0.")

    required_columns = {
        "user_id",
        "anime_id",
        "updated_at",
    }

    missing_columns = required_columns.difference(interactions.columns)

    if missing_columns:
        raise ValueError(
            f"В interactions отсутствуют столбцы: {sorted(missing_columns)}."
        )

    if interactions.empty:
        empty = interactions.copy()

        return InteractionSplit(
            train=empty.copy(),
            validation=empty.copy(),
            test=empty.copy(),
            evaluation_user_ids=frozenset(),
            train_only_user_ids=frozenset(),
        )

    ordered = interactions.sort_values(
        by=[
            "user_id",
            "updated_at",
            "anime_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    evaluation_user_ids: set[int] = set()
    train_only_user_ids: set[int] = set()

    for user_id, user_history in ordered.groupby(
        "user_id",
        sort=False,
    ):
        normalized_user_id = int(user_id)
        history_size = len(user_history)

        if history_size < min_interactions_per_user:
            train_parts.append(user_history)
            train_only_user_ids.add(normalized_user_id)
            continue

        test_size = int(history_size * test_fraction)

        validation_size = int(history_size * validation_fraction)

        test_start = history_size - test_size
        validation_start = test_start - validation_size

        can_split = test_size > 0 and validation_size > 0 and validation_start > 0

        if not can_split:
            train_parts.append(user_history)
            train_only_user_ids.add(normalized_user_id)
            continue

        train_parts.append(user_history.iloc[:validation_start])

        validation_parts.append(user_history.iloc[validation_start:test_start])

        test_parts.append(user_history.iloc[test_start:])

        evaluation_user_ids.add(normalized_user_id)

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    if validation_parts:
        validation = pd.concat(
            validation_parts,
            ignore_index=True,
        )
    else:
        validation = ordered.iloc[0:0].copy()

    if test_parts:
        test = pd.concat(
            test_parts,
            ignore_index=True,
        )
    else:
        test = ordered.iloc[0:0].copy()

    return InteractionSplit(
        train=train,
        validation=validation,
        test=test,
        evaluation_user_ids=frozenset(evaluation_user_ids),
        train_only_user_ids=frozenset(train_only_user_ids),
    )
