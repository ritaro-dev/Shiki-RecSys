from decimal import Decimal

import pandas as pd
import pytest

from shiki_recsys.preprocessing.catalog import (
    CATALOG_COLUMNS,
    prepare_catalog,
)


def _make_catalog_row(
    *,
    anime_id: int = 1,
) -> dict[str, object]:
    return {
        "id": anime_id,
        "name": "  Cowboy Bebop  ",
        "russian_name": "  Ковбой Бибоп  ",
        "kind": "  tv  ",
        "status": "  released  ",
        "score": Decimal("8.75"),
        "score_std": Decimal("0.8421"),
        "episodes": 26,
        "duration": 24,
        "rating": "  r_plus  ",
        "genres": [
            "Action",
            " Drama ",
            "Action",
            "",
            None,
        ],
        "studios": [
            "Sunrise",
            " Sunrise ",
        ],
        "stat_completed": 1000,
        "stat_dropped": 100,
        "stat_watching": 200,
        "stat_planned": 300,
    }


def test_prepare_catalog_normalizes_catalog_rows():
    rows = [
        _make_catalog_row(anime_id=2),
        {
            **_make_catalog_row(anime_id=1),
            "russian_name": "",
            "kind": None,
            "status": " ",
            "rating": None,
            "genres": None,
            "studios": [],
            "score": None,
            "score_std": None,
            "episodes": None,
        },
    ]

    result = prepare_catalog(rows)

    assert result.columns.tolist() == list(CATALOG_COLUMNS)
    assert result["anime_id"].tolist() == [1, 2]

    first_row = result.iloc[0]
    second_row = result.iloc[1]

    assert first_row["name"] == "Cowboy Bebop"
    assert pd.isna(first_row["russian_name"])
    assert pd.isna(first_row["kind"])
    assert pd.isna(first_row["status"])
    assert pd.isna(first_row["rating"])
    assert pd.isna(first_row["score"])
    assert pd.isna(first_row["score_std"])
    assert pd.isna(first_row["episodes"])
    assert first_row["genres"] == ()
    assert first_row["studios"] == ()

    assert second_row["russian_name"] == "Ковбой Бибоп"
    assert second_row["kind"] == "tv"
    assert second_row["status"] == "released"
    assert second_row["rating"] == "r_plus"
    assert second_row["genres"] == (
        "Action",
        "Drama",
    )
    assert second_row["studios"] == ("Sunrise",)


def test_prepare_catalog_assigns_canonical_dtypes():
    result = prepare_catalog([_make_catalog_row()])

    assert str(result["anime_id"].dtype) == "int64"
    assert str(result["name"].dtype) == "string"
    assert str(result["russian_name"].dtype) == "string"
    assert str(result["kind"].dtype) == "string"
    assert str(result["status"].dtype) == "string"
    assert str(result["rating"].dtype) == "string"
    assert str(result["score"].dtype) == "float32"
    assert str(result["score_std"].dtype) == "float32"
    assert str(result["episodes"].dtype) == "Int64"
    assert str(result["duration"].dtype) == "Int64"
    assert str(result["stat_completed"].dtype) == "Int64"
    assert str(result["stat_dropped"].dtype) == "Int64"
    assert str(result["stat_watching"].dtype) == "Int64"
    assert str(result["stat_planned"].dtype) == "Int64"


def test_prepare_catalog_returns_typed_empty_frame():
    result = prepare_catalog([])

    assert result.empty
    assert result.columns.tolist() == list(CATALOG_COLUMNS)
    assert str(result["anime_id"].dtype) == "int64"
    assert str(result["name"].dtype) == "string"
    assert str(result["score"].dtype) == "float32"
    assert str(result["episodes"].dtype) == "Int64"


def test_prepare_catalog_rejects_missing_columns():
    row = _make_catalog_row()
    del row["genres"]

    with pytest.raises(
        ValueError,
        match="отсутствуют столбцы",
    ):
        prepare_catalog([row])


@pytest.mark.parametrize(
    "anime_id",
    [
        0,
        -1,
    ],
)
def test_prepare_catalog_rejects_invalid_anime_id(
    anime_id: int,
):
    with pytest.raises(
        ValueError,
        match="anime_id должны быть больше 0",
    ):
        prepare_catalog([_make_catalog_row(anime_id=anime_id)])


def test_prepare_catalog_rejects_duplicate_anime_ids():
    rows = [
        _make_catalog_row(anime_id=1),
        _make_catalog_row(anime_id=1),
    ]

    with pytest.raises(
        ValueError,
        match="повторяющиеся anime_id",
    ):
        prepare_catalog(rows)


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "   ",
    ],
)
def test_prepare_catalog_rejects_empty_name(
    name: object,
):
    row = _make_catalog_row()
    row["name"] = name

    with pytest.raises(
        ValueError,
        match="непустое name",
    ):
        prepare_catalog([row])


@pytest.mark.parametrize(
    "invalid_value",
    [
        "Action",
        10,
        {"Action"},
    ],
)
def test_prepare_catalog_rejects_invalid_collection(
    invalid_value: object,
):
    row = _make_catalog_row()
    row["genres"] = invalid_value

    with pytest.raises(
        TypeError,
        match="Значение списочного признака",
    ):
        prepare_catalog([row])
