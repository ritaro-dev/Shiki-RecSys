import numpy as np
import pandas as pd
import pytest

from shiki_recsys.features.content_items import (
    _bucket_duration,
    _bucket_episodes,
    _build_binary_feature_matrix,
    build_content_item_features,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (pd.NA, "unknown"),
        (0, "unknown"),
        (1, "1"),
        (2, "2-6"),
        (6, "2-6"),
        (7, "7-13"),
        (13, "7-13"),
        (14, "14-26"),
        (26, "14-26"),
        (27, "27-52"),
        (52, "27-52"),
        (53, "53-100"),
        (100, "53-100"),
        (101, "101+"),
    ],
)
def test_bucket_episodes_assigns_expected_bucket(
    value: object,
    expected: str,
) -> None:
    assert _bucket_episodes(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (pd.NA, "unknown"),
        (0, "unknown"),
        (1, "1-5"),
        (5, "1-5"),
        (6, "6-15"),
        (15, "6-15"),
        (16, "16-30"),
        (30, "16-30"),
        (31, "31-60"),
        (60, "31-60"),
        (61, "61-120"),
        (120, "61-120"),
        (121, "121+"),
    ],
)
def test_bucket_duration_assigns_expected_bucket(
    value: object,
    expected: str,
) -> None:
    assert _bucket_duration(value) == expected


def test_build_binary_feature_matrix_encodes_content_features() -> None:
    catalog = pd.DataFrame(
        {
            "genres": [
                ("Action", "Comedy"),
                ("Comedy",),
                (),
            ],
            "studios": [
                ("Bones",),
                ("Madhouse",),
                (),
            ],
            "kind": pd.Series(
                ["tv", "movie", pd.NA],
                dtype="string",
            ),
            "rating": pd.Series(
                ["pg_13", "r", pd.NA],
                dtype="string",
            ),
            "episodes": pd.Series(
                [12, 1, pd.NA],
                dtype="Int64",
            ),
            "duration": pd.Series(
                [24, 90, pd.NA],
                dtype="Int64",
            ),
        }
    )

    result = _build_binary_feature_matrix(catalog)

    assert result.shape[0] == 3
    assert result.dtype == np.float32
    assert result.format == "csr"
    assert set(np.unique(result.data)).issubset({1.0})
    assert result.getnnz(axis=1).tolist() == [7, 6, 4]


def test_build_content_item_features_builds_normalized_tfidf_matrix() -> None:
    catalog = pd.DataFrame(
        {
            "anime_id": [30, 10, 20],
            "genres": [
                ("Action", "Comedy"),
                ("Comedy",),
                ("Drama",),
            ],
            "studios": [
                ("Bones",),
                ("Madhouse",),
                ("Bones",),
            ],
            "kind": pd.Series(
                ["tv", "movie", "tv"],
                dtype="string",
            ),
            "rating": pd.Series(
                ["pg_13", "r", "pg_13"],
                dtype="string",
            ),
            "episodes": pd.Series(
                [12, 1, 24],
                dtype="Int64",
            ),
            "duration": pd.Series(
                [24, 90, 24],
                dtype="Int64",
            ),
        }
    )

    result = build_content_item_features(catalog)

    row_norms = np.sqrt(
        result.item_feature_matrix.multiply(result.item_feature_matrix).sum(axis=1)
    ).A1

    assert result.item_feature_matrix.shape[0] == 3
    assert result.item_feature_matrix.format == "csr"
    assert result.item_feature_matrix.dtype == np.float32

    assert result.raw_anime_ids.tolist() == [30, 10, 20]
    assert result.anime_to_inner == {
        30: 0,
        10: 1,
        20: 2,
    }

    np.testing.assert_allclose(
        row_norms,
        np.ones(3),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "missing_column",
    [
        "anime_id",
        "genres",
        "studios",
        "kind",
        "rating",
        "episodes",
        "duration",
    ],
)
def test_build_content_item_features_rejects_missing_columns(
    missing_column: str,
) -> None:
    catalog = pd.DataFrame(
        {
            "anime_id": [10],
            "genres": [("Action",)],
            "studios": [("Bones",)],
            "kind": pd.Series(["tv"], dtype="string"),
            "rating": pd.Series(["pg_13"], dtype="string"),
            "episodes": pd.Series([12], dtype="Int64"),
            "duration": pd.Series([24], dtype="Int64"),
        }
    ).drop(columns=missing_column)

    with pytest.raises(
        ValueError,
        match=missing_column,
    ):
        build_content_item_features(catalog)


def test_build_content_item_features_rejects_empty_catalog() -> None:
    catalog = pd.DataFrame(
        {
            "anime_id": pd.Series(dtype="int64"),
            "genres": pd.Series(dtype="object"),
            "studios": pd.Series(dtype="object"),
            "kind": pd.Series(dtype="string"),
            "rating": pd.Series(dtype="string"),
            "episodes": pd.Series(dtype="Int64"),
            "duration": pd.Series(dtype="Int64"),
        }
    )

    with pytest.raises(
        ValueError,
        match="не должен быть пустым",
    ):
        build_content_item_features(catalog)
