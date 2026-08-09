from collections.abc import Mapping

import pandas as pd


def attach_ranker_target(
    candidate_features: pd.DataFrame,
    positive_items_by_user: Mapping[int, set[int]],
) -> pd.DataFrame:
    """
    Добавляет бинарный target к кандидатам ranker-а.

    Args:
        candidate_features: Признаки кандидатов ranker-а.
        positive_items_by_user: Положительные anime_id пользователей.

    Returns:
        Копию признаков кандидатов с бинарным target.
    """
    result = candidate_features.copy()

    result["target"] = pd.Series(
        (
            int(
                int(anime_id)
                in positive_items_by_user.get(
                    int(user_id),
                    set(),
                )
            )
            for user_id, anime_id in zip(
                result["user_id"],
                result["anime_id"],
                strict=True,
            )
        ),
        index=result.index,
        dtype="int8",
    )

    return result
