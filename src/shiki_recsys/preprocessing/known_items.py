import pandas as pd


def build_known_items(
    interactions: pd.DataFrame,
) -> dict[int, set[int]]:
    """
    Формирует множества известных объектов для пользователей.

    Args:
        interactions: Набор взаимодействий пользователей.

    Returns:
        Словарь множеств anime_id, сгруппированных по user_id.

    Raises:
        ValueError: Если отсутствуют необходимые столбцы.
    """

    required_columns = {
        "user_id",
        "anime_id",
    }

    missing_columns = required_columns.difference(interactions.columns)

    if missing_columns:
        raise ValueError(
            f"В interactions отсутствуют столбцы: {sorted(missing_columns)}."
        )

    return {
        int(user_id): set(group["anime_id"].astype(int))
        for user_id, group in interactions.groupby(
            "user_id",
            sort=False,
        )
    }
