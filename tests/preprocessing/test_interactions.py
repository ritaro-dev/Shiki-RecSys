from datetime import datetime, timezone

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from shiki_recsys.preprocessing.interactions import (
    INTERACTION_COLUMNS,
    prepare_interactions,
    select_explicit_interactions,
)


def test_prepare_interactions_converts_types():
    rows = [
        {
            "user_id": 2,
            "anime_id": 30,
            "rating": 0,
            "status": " watching ",
            "updated_at": "2026-03-01T12:00:00Z",
        },
        {
            "user_id": 1,
            "anime_id": 20,
            "rating": 9,
            "status": "completed",
            "updated_at": "2026-02-01T12:00:00Z",
        },
        {
            "user_id": 1,
            "anime_id": 10,
            "rating": 8,
            "status": "completed",
            "updated_at": "2026-01-01T12:00:00Z",
        },
    ]

    result = prepare_interactions(rows)

    assert list(result.columns) == list(INTERACTION_COLUMNS)

    assert result["status"].tolist() == [
        "watching",
        "completed",
        "completed",
    ]

    assert result["user_id"].dtype == "int64"
    assert result["anime_id"].dtype == "int64"
    assert result["rating"].dtype == "float32"

    assert isinstance(
        result["updated_at"].dtype,
        pd.DatetimeTZDtype,
    )
    assert str(result["updated_at"].dt.tz) == "UTC"


def test_prepare_interactions_returns_typed_empty_frame():
    result = prepare_interactions([])

    assert result.empty
    assert list(result.columns) == list(INTERACTION_COLUMNS)

    assert result["user_id"].dtype == "int64"
    assert result["anime_id"].dtype == "int64"
    assert result["rating"].dtype == "float32"
    assert result["status"].dtype == "string"
    assert str(result["updated_at"].dtype) == ("datetime64[ns, UTC]")


def test_prepare_interactions_rejects_missing_column():
    rows = [
        {
            "user_id": 1,
            "anime_id": 10,
            "rating": 8,
            "status": "completed",
        },
    ]

    with pytest.raises(
        ValueError,
        match="отсутствуют столбцы",
    ):
        prepare_interactions(rows)


def test_prepare_interactions_rejects_null_values():
    rows = [
        {
            "user_id": 1,
            "anime_id": 10,
            "rating": 8,
            "status": "completed",
            "updated_at": None,
        },
    ]

    with pytest.raises(
        ValueError,
        match="пропущенные значения",
    ):
        prepare_interactions(rows)


def test_prepare_interactions_rejects_invalid_types():
    rows = [
        {
            "user_id": "unknown",
            "anime_id": 10,
            "rating": 8,
            "status": "completed",
            "updated_at": datetime.now(timezone.utc),
        },
    ]

    with pytest.raises(
        ValueError,
        match="ожидаемым типам данных",
    ):
        prepare_interactions(rows)


def test_select_explicit_interactions_keeps_only_positive_ratings() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [
                1,
                1,
                2,
                2,
                3,
            ],
            "anime_id": [
                10,
                20,
                30,
                40,
                50,
            ],
            "rating": [
                0.0,
                8.0,
                3.0,
                0.0,
                10.0,
            ],
            "status": [
                "watching",
                "completed",
                "dropped",
                "planned",
                "completed",
            ],
        }
    )

    result = select_explicit_interactions(interactions)

    expected = pd.DataFrame(
        {
            "user_id": [
                1,
                2,
                3,
            ],
            "anime_id": [
                20,
                30,
                50,
            ],
            "rating": [
                8.0,
                3.0,
                10.0,
            ],
            "status": [
                "completed",
                "dropped",
                "completed",
            ],
        }
    )

    assert_frame_equal(
        result,
        expected,
    )


def test_select_explicit_interactions_preserves_all_columns() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1],
            "anime_id": [10, 20],
            "rating": [0.0, 9.0],
            "status": ["watching", "completed"],
            "updated_at": pd.to_datetime(
                [
                    "2026-01-01T10:00:00Z",
                    "2026-01-02T10:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    result = select_explicit_interactions(interactions)

    assert list(result.columns) == list(interactions.columns)

    assert (
        result.loc[
            0,
            "updated_at",
        ]
        == interactions.loc[
            1,
            "updated_at",
        ]
    )


def test_select_explicit_interactions_returns_empty_frame_when_no_ratings() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": pd.Series(
                [1, 2],
                dtype="int64",
            ),
            "anime_id": pd.Series(
                [10, 20],
                dtype="int64",
            ),
            "rating": pd.Series(
                [0.0, 0.0],
                dtype="float32",
            ),
            "status": pd.Series(
                ["watching", "planned"],
                dtype="string",
            ),
        }
    )

    result = select_explicit_interactions(interactions)

    expected = interactions.iloc[0:0].copy().reset_index(drop=True)

    assert_frame_equal(
        result,
        expected,
    )


def test_select_explicit_interactions_rejects_missing_rating_column() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
        }
    )

    with pytest.raises(
        ValueError,
        match="отсутствует столбец rating",
    ):
        select_explicit_interactions(interactions)


def test_select_explicit_interactions_returns_independent_copy() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 2],
            "anime_id": [10, 20],
            "rating": [8.0, 9.0],
        }
    )

    result = select_explicit_interactions(interactions)

    result.loc[
        0,
        "anime_id",
    ] = 999

    assert (
        interactions.loc[
            0,
            "anime_id",
        ]
        == 10
    )
