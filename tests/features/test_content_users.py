import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix

from shiki_recsys.features.content_items import ContentItemFeatures
from shiki_recsys.features.content_users import build_content_user_profiles


def test_build_content_user_profiles_uses_recent_positive_items() -> None:
    item_features = ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.eye(
                4,
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [10, 20, 30, 40],
            dtype=np.int64,
        ),
        anime_to_inner={
            10: 0,
            20: 1,
            30: 2,
            40: 3,
        },
    )

    train_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2],
            "anime_id": [10, 20, 30, 40, 40],
            "rating": pd.Series(
                [10, 8, 9, 7, 8],
                dtype="float32",
            ),
            "updated_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                    "2026-03-01",
                    "2026-04-01",
                    "2026-01-15",
                ],
                utc=True,
            ),
        }
    )

    result = build_content_user_profiles(
        train_interactions,
        item_features,
        relevance_threshold=8,
        max_positive_items=2,
    )

    expected = np.array(
        [
            [
                0.0,
                1 / np.sqrt(2),
                1 / np.sqrt(2),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float32,
    )

    assert result.raw_user_ids.tolist() == [1, 2]
    assert result.user_to_inner == {
        1: 0,
        2: 1,
    }
    assert result.user_profile_matrix.format == "csr"
    assert result.user_profile_matrix.dtype == np.float32

    np.testing.assert_allclose(
        result.user_profile_matrix.toarray(),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_build_content_user_profiles_returns_empty_profiles_without_positive_items() -> (
    None
):
    item_features = ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.eye(
                2,
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [10, 20],
            dtype=np.int64,
        ),
        anime_to_inner={
            10: 0,
            20: 1,
        },
    )

    train_interactions = pd.DataFrame(
        {
            "user_id": [1, 2],
            "anime_id": [10, 20],
            "rating": pd.Series(
                [7, 0],
                dtype="float32",
            ),
            "updated_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ],
                utc=True,
            ),
        }
    )

    result = build_content_user_profiles(
        train_interactions,
        item_features,
        relevance_threshold=8,
        max_positive_items=50,
    )

    assert result.user_profile_matrix.shape == (0, 2)
    assert result.user_profile_matrix.format == "csr"
    assert result.user_profile_matrix.dtype == np.float32
    assert result.raw_user_ids.tolist() == []
    assert result.user_to_inner == {}


def test_build_content_user_profiles_rejects_positive_item_missing_from_mapping() -> (
    None
):
    item_features = ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.eye(
                1,
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [10],
            dtype=np.int64,
        ),
        anime_to_inner={
            10: 0,
        },
    )

    train_interactions = pd.DataFrame(
        {
            "user_id": [1, 1],
            "anime_id": [10, 20],
            "rating": pd.Series(
                [8, 9],
                dtype="float32",
            ),
            "updated_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ],
                utc=True,
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="20",
    ):
        build_content_user_profiles(
            train_interactions,
            item_features,
            relevance_threshold=8,
            max_positive_items=50,
        )


@pytest.mark.parametrize(
    "missing_column",
    [
        "user_id",
        "anime_id",
        "rating",
        "updated_at",
    ],
)
def test_build_content_user_profiles_rejects_missing_columns(
    missing_column: str,
) -> None:
    item_features = ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.eye(
                1,
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [10],
            dtype=np.int64,
        ),
        anime_to_inner={
            10: 0,
        },
    )

    train_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "rating": pd.Series([8], dtype="float32"),
            "updated_at": pd.to_datetime(
                ["2026-01-01"],
                utc=True,
            ),
        }
    ).drop(columns=missing_column)

    with pytest.raises(
        ValueError,
        match=missing_column,
    ):
        build_content_user_profiles(
            train_interactions,
            item_features,
            relevance_threshold=8,
            max_positive_items=50,
        )


def test_build_content_user_profiles_rejects_empty_input() -> None:
    item_features = ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.eye(
                1,
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [10],
            dtype=np.int64,
        ),
        anime_to_inner={
            10: 0,
        },
    )

    train_interactions = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "anime_id": pd.Series(dtype="int64"),
            "rating": pd.Series(dtype="float32"),
            "updated_at": pd.Series(
                dtype="datetime64[ns, UTC]",
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="не должен быть пустым",
    ):
        build_content_user_profiles(
            train_interactions,
            item_features,
            relevance_threshold=8,
            max_positive_items=50,
        )


@pytest.mark.parametrize(
    "relevance_threshold",
    [
        0,
        -1,
        10.1,
        math.nan,
        math.inf,
    ],
)
def test_build_content_user_profiles_rejects_invalid_relevance_threshold(
    relevance_threshold: float,
) -> None:
    item_features = ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.eye(
                1,
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [10],
            dtype=np.int64,
        ),
        anime_to_inner={
            10: 0,
        },
    )

    train_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "rating": pd.Series([8], dtype="float32"),
            "updated_at": pd.to_datetime(
                ["2026-01-01"],
                utc=True,
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="relevance_threshold",
    ):
        build_content_user_profiles(
            train_interactions,
            item_features,
            relevance_threshold=relevance_threshold,
            max_positive_items=50,
        )


@pytest.mark.parametrize(
    "max_positive_items",
    [
        0,
        -1,
    ],
)
def test_build_content_user_profiles_rejects_invalid_max_positive_items(
    max_positive_items: int,
) -> None:
    item_features = ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.eye(
                1,
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [10],
            dtype=np.int64,
        ),
        anime_to_inner={
            10: 0,
        },
    )

    train_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "rating": pd.Series([8], dtype="float32"),
            "updated_at": pd.to_datetime(
                ["2026-01-01"],
                utc=True,
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="max_positive_items",
    ):
        build_content_user_profiles(
            train_interactions,
            item_features,
            relevance_threshold=8,
            max_positive_items=max_positive_items,
        )


def test_build_content_user_profiles_does_not_modify_input() -> None:
    item_features = ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.eye(
                2,
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [10, 20],
            dtype=np.int64,
        ),
        anime_to_inner={
            10: 0,
            20: 1,
        },
    )

    train_interactions = pd.DataFrame(
        {
            "user_id": [1, 1],
            "anime_id": [10, 20],
            "rating": pd.Series(
                [8, 9],
                dtype="float32",
            ),
            "updated_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ],
                utc=True,
            ),
        }
    )
    original = train_interactions.copy(deep=True)

    build_content_user_profiles(
        train_interactions,
        item_features,
        relevance_threshold=8,
        max_positive_items=50,
    )

    assert_frame_equal(
        train_interactions,
        original,
    )
