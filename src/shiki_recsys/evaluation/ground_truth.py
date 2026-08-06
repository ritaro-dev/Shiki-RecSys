import math

import pandas as pd


def build_positive_items_by_user(
    interactions: pd.DataFrame,
    *,
    positive_rating_threshold: float,
) -> dict[int, set[int]]:
    """
    Формирует положительные целевые объекты пользователей.

    Args:
        interactions: Взаимодействия из validation или test.
        positive_rating_threshold: Минимальная положительная оценка.

    Returns:
        Положительные anime_id, сгруппированные по user_id.

    Raises:
        ValueError: Если отсутствуют обязательные столбцы
            или порог положительной оценки некорректен.
    """

    required_columns = {
        "user_id",
        "anime_id",
        "rating",
    }

    missing_columns = required_columns.difference(interactions.columns)

    if missing_columns:
        raise ValueError(
            f"В interactions отсутствуют столбцы: {sorted(missing_columns)}."
        )

    if (
        not math.isfinite(positive_rating_threshold)
        or not 0 < positive_rating_threshold <= 10
    ):
        raise ValueError(
            "positive_rating_threshold должен быть конечным "
            "числом в диапазоне от 0 до 10."
        )

    positive_interactions = interactions.loc[
        interactions["rating"] >= positive_rating_threshold,
        [
            "user_id",
            "anime_id",
        ],
    ]

    return {
        int(user_id): set(group["anime_id"].astype(int))
        for user_id, group in positive_interactions.groupby(
            "user_id",
            sort=False,
        )
    }
