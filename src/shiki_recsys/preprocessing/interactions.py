from collections.abc import Iterable, Mapping

import pandas as pd

INTERACTION_COLUMNS = (
    "user_id",
    "anime_id",
    "rating",
    "status",
    "updated_at",
)


def _empty_interactions_frame() -> pd.DataFrame:
    """
    Создаёт пустой DataFrame с каноническими
    типами столбцов взаимодействий.
    """

    return pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "anime_id": pd.Series(dtype="int64"),
            "rating": pd.Series(dtype="float32"),
            "status": pd.Series(dtype="string"),
            "updated_at": pd.Series(
                dtype="datetime64[ns, UTC]",
            ),
        }
    )


def prepare_interactions(
    rows: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    """
    Преобразует строки взаимодействий из
    database-слоя в канонический DataFrame.

    Функция:
    - проверяет обязательные столбцы;
    - запрещает пропущенные значения;
    - приводит типы.
    """

    frame = pd.DataFrame.from_records(rows)

    if frame.empty:
        return _empty_interactions_frame()

    missing_columns = [
        column for column in INTERACTION_COLUMNS if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"В данных взаимодействий отсутствуют столбцы: {missing_columns}."
        )

    frame = frame.loc[
        :,
        list(INTERACTION_COLUMNS),
    ].copy()

    columns_with_nulls = [
        column for column in INTERACTION_COLUMNS if frame[column].isna().any()
    ]

    if columns_with_nulls:
        raise ValueError(
            "В данных взаимодействий обнаружены "
            "пропущенные значения в столбцах: "
            f"{columns_with_nulls}."
        )

    try:
        frame["user_id"] = pd.to_numeric(
            frame["user_id"],
            errors="raise",
        ).astype("int64")

        frame["anime_id"] = pd.to_numeric(
            frame["anime_id"],
            errors="raise",
        ).astype("int64")

        frame["rating"] = pd.to_numeric(
            frame["rating"],
            errors="raise",
        ).astype("float32")

        frame["updated_at"] = pd.to_datetime(
            frame["updated_at"],
            errors="raise",
            utc=True,
        )

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Не удалось привести взаимодействия к ожидаемым типам данных."
        ) from exc

    frame["status"] = frame["status"].astype("string").str.strip()

    return frame.reset_index(drop=True)


def select_explicit_interactions(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Выбирает взаимодействия с явной пользовательской оценкой.

    Args:
        interactions: Подготовленная таблица взаимодействий.

    Returns:
        Взаимодействия с rating больше 0.

    Raises:
        ValueError: Если отсутствует столбец rating.
    """

    if "rating" not in interactions.columns:
        raise ValueError("В interactions отсутствует столбец rating.")

    return interactions.loc[interactions["rating"] > 0].copy().reset_index(drop=True)
