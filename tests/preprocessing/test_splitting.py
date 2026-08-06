import pandas as pd
import pytest

from shiki_recsys.preprocessing.splitting import (
    chronological_split,
)

VALIDATION_FRACTION = 0.1
TEST_FRACTION = 0.1
MIN_INTERACTIONS_PER_USER = 10


def _make_interactions(
    *,
    user_id: int,
    interaction_count: int,
    first_anime_id: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [user_id] * interaction_count,
            "anime_id": range(
                first_anime_id,
                first_anime_id + interaction_count,
            ),
            "rating": [8.0] * interaction_count,
            "status": ["completed"] * interaction_count,
            "updated_at": pd.date_range(
                start="2025-01-01",
                periods=interaction_count,
                freq="D",
                tz="UTC",
            ),
        }
    )


def test_chronological_split_separates_each_user_history():
    user_1 = _make_interactions(
        user_id=1,
        interaction_count=10,
        first_anime_id=100,
    )
    user_2 = _make_interactions(
        user_id=2,
        interaction_count=23,
        first_anime_id=200,
    )

    interactions = (
        pd.concat(
            [user_1, user_2],
            ignore_index=True,
        )
        .sample(
            frac=1,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    result = chronological_split(
        interactions,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
        min_interactions_per_user=MIN_INTERACTIONS_PER_USER,
    )

    user_1_train = result.train[result.train["user_id"] == 1]
    user_1_validation = result.validation[result.validation["user_id"] == 1]
    user_1_test = result.test[result.test["user_id"] == 1]

    assert user_1_train["anime_id"].tolist() == list(range(100, 108))
    assert user_1_validation["anime_id"].tolist() == [108]
    assert user_1_test["anime_id"].tolist() == [109]

    user_2_train = result.train[result.train["user_id"] == 2]
    user_2_validation = result.validation[result.validation["user_id"] == 2]
    user_2_test = result.test[result.test["user_id"] == 2]

    assert user_2_train["anime_id"].tolist() == list(range(200, 219))
    assert user_2_validation["anime_id"].tolist() == [
        219,
        220,
    ]
    assert user_2_test["anime_id"].tolist() == [
        221,
        222,
    ]

    assert result.evaluation_user_ids == frozenset({1, 2})
    assert result.train_only_user_ids == frozenset()

    assert len(result.train) + len(result.validation) + len(result.test) == len(
        interactions
    )


def test_chronological_split_puts_short_history_only_in_train():
    interactions = _make_interactions(
        user_id=3,
        interaction_count=9,
        first_anime_id=300,
    )

    result = chronological_split(
        interactions,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
        min_interactions_per_user=MIN_INTERACTIONS_PER_USER,
    )

    assert result.train["anime_id"].tolist() == list(range(300, 309))
    assert result.validation.empty
    assert result.test.empty

    assert result.evaluation_user_ids == frozenset()
    assert result.train_only_user_ids == frozenset({3})


def test_chronological_split_puts_unsplittable_history_only_in_train():
    interactions = _make_interactions(
        user_id=4,
        interaction_count=10,
        first_anime_id=400,
    )

    result = chronological_split(
        interactions,
        validation_fraction=0.01,
        test_fraction=0.01,
        min_interactions_per_user=1,
    )

    assert len(result.train) == 10
    assert result.validation.empty
    assert result.test.empty

    assert result.evaluation_user_ids == frozenset()
    assert result.train_only_user_ids == frozenset({4})


def test_chronological_split_preserves_empty_frame_structure():
    interactions = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "anime_id": pd.Series(dtype="int64"),
            "rating": pd.Series(dtype="float32"),
            "status": pd.Series(dtype="string"),
            "updated_at": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )

    result = chronological_split(
        interactions,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
        min_interactions_per_user=MIN_INTERACTIONS_PER_USER,
    )

    for part in (
        result.train,
        result.validation,
        result.test,
    ):
        assert part.empty
        assert part.columns.tolist() == interactions.columns.tolist()
        assert part.dtypes.to_dict() == interactions.dtypes.to_dict()

    assert result.evaluation_user_ids == frozenset()
    assert result.train_only_user_ids == frozenset()


@pytest.mark.parametrize(
    (
        "validation_fraction",
        "test_fraction",
    ),
    [
        (0.0, 0.1),
        (-0.1, 0.1),
        (1.0, 0.1),
        (0.1, 0.0),
        (0.1, -0.1),
        (0.1, 1.0),
        (0.5, 0.5),
        (0.7, 0.4),
    ],
)
def test_chronological_split_rejects_invalid_fractions(
    validation_fraction: float,
    test_fraction: float,
):
    interactions = _make_interactions(
        user_id=1,
        interaction_count=10,
        first_anime_id=100,
    )

    with pytest.raises(ValueError):
        chronological_split(
            interactions,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            min_interactions_per_user=MIN_INTERACTIONS_PER_USER,
        )


@pytest.mark.parametrize(
    "min_interactions_per_user",
    [
        0,
        -1,
    ],
)
def test_chronological_split_rejects_invalid_minimum(
    min_interactions_per_user: int,
):
    interactions = _make_interactions(
        user_id=1,
        interaction_count=10,
        first_anime_id=100,
    )

    with pytest.raises(
        ValueError,
        match="min_interactions_per_user",
    ):
        chronological_split(
            interactions,
            validation_fraction=VALIDATION_FRACTION,
            test_fraction=TEST_FRACTION,
            min_interactions_per_user=min_interactions_per_user,
        )


def test_chronological_split_rejects_missing_columns():
    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [100],
        }
    )

    with pytest.raises(
        ValueError,
        match="отсутствуют столбцы",
    ):
        chronological_split(
            interactions,
            validation_fraction=VALIDATION_FRACTION,
            test_fraction=TEST_FRACTION,
            min_interactions_per_user=MIN_INTERACTIONS_PER_USER,
        )
