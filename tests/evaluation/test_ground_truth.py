import math

import pandas as pd
import pytest

from shiki_recsys.evaluation.ground_truth import (
    build_positive_items_by_user,
)


def test_build_positive_items_by_user_groups_positive_items() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [
                1,
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
                60,
            ],
            "rating": [
                9.0,
                8.0,
                7.0,
                10.0,
                5.0,
                0.0,
            ],
        }
    )

    result = build_positive_items_by_user(
        interactions,
        positive_rating_threshold=8,
    )

    assert result == {
        1: {10, 20},
        2: {40},
    }


def test_build_positive_items_by_user_includes_threshold_rating() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1],
            "anime_id": [10, 20],
            "rating": [8.0, 7.9],
        }
    )

    result = build_positive_items_by_user(
        interactions,
        positive_rating_threshold=8,
    )

    assert result == {
        1: {10},
    }


def test_build_positive_items_by_user_removes_duplicate_items() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "anime_id": [10, 10, 20],
            "rating": [9.0, 10.0, 8.0],
        }
    )

    result = build_positive_items_by_user(
        interactions,
        positive_rating_threshold=8,
    )

    assert result == {
        1: {10, 20},
    }


def test_build_positive_items_by_user_omits_users_without_positive_items() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 2],
            "anime_id": [10, 20],
            "rating": [7.0, 0.0],
        }
    )

    result = build_positive_items_by_user(
        interactions,
        positive_rating_threshold=8,
    )

    assert result == {}


def test_build_positive_items_by_user_returns_empty_mapping_for_empty_frame() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "anime_id": pd.Series(dtype="int64"),
            "rating": pd.Series(dtype="float32"),
        }
    )

    result = build_positive_items_by_user(
        interactions,
        positive_rating_threshold=8,
    )

    assert result == {}


@pytest.mark.parametrize(
    (
        "interactions",
        "missing_column",
    ),
    [
        (
            pd.DataFrame(
                {
                    "anime_id": [10],
                    "rating": [8.0],
                }
            ),
            "user_id",
        ),
        (
            pd.DataFrame(
                {
                    "user_id": [1],
                    "rating": [8.0],
                }
            ),
            "anime_id",
        ),
        (
            pd.DataFrame(
                {
                    "user_id": [1],
                    "anime_id": [10],
                }
            ),
            "rating",
        ),
    ],
)
def test_build_positive_items_by_user_rejects_missing_columns(
    interactions: pd.DataFrame,
    missing_column: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=missing_column,
    ):
        build_positive_items_by_user(
            interactions,
            positive_rating_threshold=8,
        )


@pytest.mark.parametrize(
    "positive_rating_threshold",
    [
        0,
        -1,
        10.1,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_build_positive_items_by_user_rejects_invalid_threshold(
    positive_rating_threshold: float,
) -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "rating": [8.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="positive_rating_threshold",
    ):
        build_positive_items_by_user(
            interactions,
            positive_rating_threshold=positive_rating_threshold,
        )
