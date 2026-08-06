import math

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from shiki_recsys.features.als_signals import (
    build_als_signed_interactions,
)


def _build_signed_interactions(
    train_interactions: pd.DataFrame,
    **confidence_overrides: float,
) -> pd.DataFrame:
    confidence_values = {
        "rating_8_10_confidence": 2.0,
        "watching_confidence": 1.0,
        "rewatching_confidence": 1.0,
        "completed_confidence": 0.5,
        "planned_confidence": 0.5,
        "on_hold_confidence": 0.0,
        "rating_4_5_confidence": -1.0,
        "rating_1_3_confidence": -2.0,
    }
    confidence_values.update(confidence_overrides)

    return build_als_signed_interactions(
        train_interactions,
        rating_8_10_confidence=confidence_values["rating_8_10_confidence"],
        watching_confidence=confidence_values["watching_confidence"],
        rewatching_confidence=confidence_values["rewatching_confidence"],
        completed_confidence=confidence_values["completed_confidence"],
        planned_confidence=confidence_values["planned_confidence"],
        on_hold_confidence=confidence_values["on_hold_confidence"],
        rating_4_5_confidence=confidence_values["rating_4_5_confidence"],
        rating_1_3_confidence=confidence_values["rating_1_3_confidence"],
    )


def test_build_als_signed_interactions_assigns_selected_signals() -> None:
    train_interactions = pd.DataFrame(
        {
            "user_id": [
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "anime_id": [
                10,
                150,
                140,
                130,
                120,
                110,
                100,
                90,
                80,
                70,
                60,
                50,
                40,
                30,
                20,
            ],
            "rating": [
                0,
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                7,
                6,
                3,
                1,
                5,
                4,
                8,
            ],
            "status": [
                "watching",
                "watching",
                "dropped",
                "on_hold",
                "planned",
                "completed",
                "rewatching",
                "watching",
                "completed",
                "watching",
                "completed",
                "completed",
                "watching",
                "completed",
                "completed",
            ],
        }
    )

    result = _build_signed_interactions(train_interactions)

    expected = pd.DataFrame(
        {
            "user_id": pd.Series(
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                ],
                dtype="int64",
            ),
            "anime_id": pd.Series(
                [
                    20,
                    30,
                    40,
                    50,
                    60,
                    90,
                    100,
                    110,
                    120,
                    150,
                    10,
                ],
                dtype="int64",
            ),
            "confidence": pd.Series(
                [
                    2.0,
                    -1.0,
                    -1.0,
                    -2.0,
                    -2.0,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                    -1.0,
                    1.0,
                ],
                dtype="float32",
            ),
        }
    )

    assert_frame_equal(
        result,
        expected,
    )


def test_build_als_signed_interactions_uses_explicit_rating_before_status() -> None:
    train_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "anime_id": [10, 20, 30],
            "rating": [9, 3, 5],
            "status": [
                "dropped",
                "completed",
                "watching",
            ],
        }
    )

    result = _build_signed_interactions(train_interactions)

    assert result["confidence"].tolist() == [
        2.0,
        -2.0,
        -1.0,
    ]


def test_build_als_signed_interactions_removes_neutral_signals() -> None:
    train_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1],
            "anime_id": [10, 20, 30, 40, 50],
            "rating": [8, 6, 7, 0, 0],
            "status": [
                "completed",
                "watching",
                "completed",
                "on_hold",
                "dropped",
            ],
        }
    )

    result = _build_signed_interactions(train_interactions)

    assert result["anime_id"].tolist() == [10]
    assert result["confidence"].tolist() == [2.0]


def test_build_als_signed_interactions_does_not_modify_input() -> None:
    train_interactions = pd.DataFrame(
        {
            "user_id": [1, 1],
            "anime_id": [10, 20],
            "rating": [8, 0],
            "status": [
                "completed",
                "watching",
            ],
        }
    )
    original = train_interactions.copy(deep=True)

    _build_signed_interactions(train_interactions)

    assert_frame_equal(
        train_interactions,
        original,
    )


@pytest.mark.parametrize(
    "missing_column",
    [
        "user_id",
        "anime_id",
        "rating",
        "status",
    ],
)
def test_build_als_signed_interactions_rejects_missing_columns(
    missing_column: str,
) -> None:
    train_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "rating": [8],
            "status": ["completed"],
        }
    ).drop(columns=missing_column)

    with pytest.raises(
        ValueError,
        match=missing_column,
    ):
        _build_signed_interactions(train_interactions)


def test_build_als_signed_interactions_rejects_empty_input() -> None:
    train_interactions = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "anime_id": pd.Series(dtype="int64"),
            "rating": pd.Series(dtype="float32"),
            "status": pd.Series(dtype="string"),
        }
    )

    with pytest.raises(
        ValueError,
        match="не должен быть пустым",
    ):
        _build_signed_interactions(train_interactions)


@pytest.mark.parametrize(
    (
        "parameter_name",
        "invalid_value",
    ),
    [
        (
            "rating_8_10_confidence",
            0.0,
        ),
        (
            "rating_8_10_confidence",
            math.inf,
        ),
        (
            "watching_confidence",
            -0.1,
        ),
        (
            "rewatching_confidence",
            -0.1,
        ),
        (
            "completed_confidence",
            -0.1,
        ),
        (
            "planned_confidence",
            -0.1,
        ),
        (
            "on_hold_confidence",
            -0.1,
        ),
        (
            "rating_4_5_confidence",
            0.1,
        ),
        (
            "rating_1_3_confidence",
            0.1,
        ),
        (
            "rating_1_3_confidence",
            math.nan,
        ),
    ],
)
def test_build_als_signed_interactions_rejects_invalid_confidences(
    parameter_name: str,
    invalid_value: float,
) -> None:
    train_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "rating": [8],
            "status": ["completed"],
        }
    )

    with pytest.raises(
        ValueError,
        match=parameter_name,
    ):
        _build_signed_interactions(
            train_interactions,
            **{
                parameter_name: invalid_value,
            },
        )


def test_build_als_signed_interactions_rejects_only_neutral_signals() -> None:
    train_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "anime_id": [10, 20, 30],
            "rating": [6, 7, 0],
            "status": [
                "completed",
                "watching",
                "dropped",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="не осталось взаимодействий",
    ):
        _build_signed_interactions(train_interactions)


def test_build_als_signed_interactions_rejects_only_negative_signals() -> None:
    train_interactions = pd.DataFrame(
        {
            "user_id": [1, 1],
            "anime_id": [10, 20],
            "rating": [5, 2],
            "status": [
                "completed",
                "completed",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="не содержат положительных сигналов",
    ):
        _build_signed_interactions(train_interactions)
