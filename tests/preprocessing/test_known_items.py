import pandas as pd
import pytest

from shiki_recsys.preprocessing.known_items import (
    build_known_items,
)


def test_build_known_items_groups_anime_by_user():
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 1],
            "anime_id": [10, 20, 30, 10],
            "rating": [8, 0, 0, 8],
        }
    )

    result = build_known_items(interactions)

    assert result == {
        1: {10, 20},
        2: {30},
    }


def test_build_known_items_returns_empty_dict():
    interactions = pd.DataFrame(
        columns=[
            "user_id",
            "anime_id",
        ]
    )

    result = build_known_items(interactions)

    assert result == {}


def test_build_known_items_rejects_missing_columns():
    interactions = pd.DataFrame(
        {
            "user_id": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="отсутствуют столбцы",
    ):
        build_known_items(interactions)
