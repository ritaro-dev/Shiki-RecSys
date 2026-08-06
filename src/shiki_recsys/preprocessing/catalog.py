from collections.abc import Iterable, Mapping

import pandas as pd

SOURCE_CATALOG_COLUMNS = (
    "id",
    "name",
    "russian_name",
    "kind",
    "status",
    "score",
    "score_std",
    "episodes",
    "duration",
    "rating",
    "genres",
    "studios",
    "stat_completed",
    "stat_dropped",
    "stat_watching",
    "stat_planned",
)

CATALOG_COLUMNS = (
    "anime_id",
    "name",
    "russian_name",
    "kind",
    "status",
    "score",
    "score_std",
    "episodes",
    "duration",
    "rating",
    "genres",
    "studios",
    "stat_completed",
    "stat_dropped",
    "stat_watching",
    "stat_planned",
)

OPTIONAL_TEXT_COLUMNS = (
    "russian_name",
    "kind",
    "status",
    "rating",
)

FLOAT_COLUMNS = (
    "score",
    "score_std",
)

NULLABLE_INTEGER_COLUMNS = (
    "episodes",
    "duration",
    "stat_completed",
    "stat_dropped",
    "stat_watching",
    "stat_planned",
)

COLLECTION_COLUMNS = (
    "genres",
    "studios",
)


def _empty_catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anime_id": pd.Series(dtype="int64"),
            "name": pd.Series(dtype="string"),
            "russian_name": pd.Series(dtype="string"),
            "kind": pd.Series(dtype="string"),
            "status": pd.Series(dtype="string"),
            "score": pd.Series(dtype="float32"),
            "score_std": pd.Series(dtype="float32"),
            "episodes": pd.Series(dtype="Int64"),
            "duration": pd.Series(dtype="Int64"),
            "rating": pd.Series(dtype="string"),
            "genres": pd.Series(dtype="object"),
            "studios": pd.Series(dtype="object"),
            "stat_completed": pd.Series(dtype="Int64"),
            "stat_dropped": pd.Series(dtype="Int64"),
            "stat_watching": pd.Series(dtype="Int64"),
            "stat_planned": pd.Series(dtype="Int64"),
        }
    )


def _normalize_text_collection(
    value: object,
) -> tuple[str, ...]:
    if value is None or value is pd.NA:
        return ()

    if isinstance(value, float) and pd.isna(value):
        return ()

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "Значение списочного признака должно быть списком, кортежем или NULL."
        )

    normalized_values: list[str] = []
    seen_values: set[str] = set()

    for item in value:
        if item is None:
            continue

        normalized_item = str(item).strip()

        if normalized_item and normalized_item not in seen_values:
            normalized_values.append(normalized_item)
            seen_values.add(normalized_item)

    return tuple(normalized_values)


def prepare_catalog(
    rows: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    """
    Формирует канонический общий каталог аниме.

    Args:
        rows: Строки каталога, полученные из хранилища.

    Returns:
        Подготовленный каталог с каноническими именами столбцов
        и типами данных.

    Raises:
        ValueError: Если отсутствуют необходимые столбцы,
            идентификаторы или названия некорректны, обнаружены
            повторяющиеся идентификаторы либо списочные признаки
            имеют некорректный формат.
    """

    catalog = pd.DataFrame(rows)

    if catalog.empty:
        return _empty_catalog_frame()

    missing_columns = set(SOURCE_CATALOG_COLUMNS).difference(catalog.columns)

    if missing_columns:
        raise ValueError(
            f"В строках каталога отсутствуют столбцы: {sorted(missing_columns)}."
        )

    catalog = (
        catalog.loc[:, SOURCE_CATALOG_COLUMNS].rename(columns={"id": "anime_id"}).copy()
    )

    if catalog["anime_id"].isna().any():
        raise ValueError("Каталог содержит пустые anime_id.")

    catalog["anime_id"] = pd.to_numeric(
        catalog["anime_id"],
        errors="raise",
    ).astype("int64")

    if (catalog["anime_id"] <= 0).any():
        raise ValueError("Все anime_id должны быть больше 0.")

    if catalog["anime_id"].duplicated().any():
        duplicate_ids = (
            catalog.loc[
                catalog["anime_id"].duplicated(keep=False),
                "anime_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(f"Каталог содержит повторяющиеся anime_id: {duplicate_ids}.")

    catalog["name"] = catalog["name"].astype("string").str.strip()

    if catalog["name"].isna().any() or catalog["name"].eq("").any():
        raise ValueError("Каждое аниме должно иметь непустое name.")

    for column in OPTIONAL_TEXT_COLUMNS:
        catalog[column] = (
            catalog[column].astype("string").str.strip().replace("", pd.NA)
        )

    for column in FLOAT_COLUMNS:
        catalog[column] = pd.to_numeric(
            catalog[column],
            errors="raise",
        ).astype("float32")

    for column in NULLABLE_INTEGER_COLUMNS:
        catalog[column] = pd.to_numeric(
            catalog[column],
            errors="raise",
        ).astype("Int64")

    for column in COLLECTION_COLUMNS:
        catalog[column] = catalog[column].map(_normalize_text_collection)

    return (
        catalog.loc[:, CATALOG_COLUMNS]
        .sort_values(
            by="anime_id",
            kind="stable",
        )
        .reset_index(drop=True)
    )
