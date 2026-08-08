import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from shiki_recsys.inference.candidate_filtering import (
    exclude_known_candidates,
)
from shiki_recsys.retrievers.common import RetrieverName


def test_exclude_known_candidates_preserves_order_and_columns() -> None:
    candidates = pd.DataFrame(
        {
            "anime_id": pd.Series(
                [10, 20, 30, 40],
                dtype="int64",
            ),
            "score": pd.Series(
                [0.9, 0.8, 0.7, 0.6],
                dtype="float64",
            ),
            "source": pd.Series(
                [
                    RetrieverName.POPULARITY.value,
                    RetrieverName.POPULARITY.value,
                    RetrieverName.POPULARITY.value,
                    RetrieverName.POPULARITY.value,
                ],
                dtype="string",
            ),
            "source_rank": pd.Series(
                [1, 2, 3, 4],
                dtype="int32",
            ),
        }
    )

    result = exclude_known_candidates(
        candidates,
        known_anime_ids={10, 30},
    )

    expected = pd.DataFrame(
        {
            "anime_id": pd.Series(
                [20, 40],
                dtype="int64",
            ),
            "score": pd.Series(
                [0.8, 0.6],
                dtype="float64",
            ),
            "source": pd.Series(
                [
                    RetrieverName.POPULARITY.value,
                    RetrieverName.POPULARITY.value,
                ],
                dtype="string",
            ),
            "source_rank": pd.Series(
                [2, 4],
                dtype="int32",
            ),
        }
    )

    assert_frame_equal(
        result,
        expected,
    )


def test_exclude_known_candidates_applies_limit_after_filtering() -> None:
    candidates = pd.DataFrame(
        {
            "anime_id": [10, 20, 30, 40, 50],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )

    result = exclude_known_candidates(
        candidates,
        known_anime_ids={10, 20},
        candidate_count=2,
    )

    expected = pd.DataFrame(
        {
            "anime_id": [30, 40],
            "score": [0.7, 0.6],
        }
    )

    assert_frame_equal(
        result,
        expected,
    )


def test_exclude_known_candidates_returns_all_candidates_for_empty_known_set() -> None:
    candidates = pd.DataFrame(
        {
            "anime_id": [10, 20],
            "score": [0.9, 0.8],
        }
    )

    result = exclude_known_candidates(
        candidates,
        known_anime_ids=set(),
    )

    assert_frame_equal(
        result,
        candidates,
    )


def test_exclude_known_candidates_returns_empty_frame_when_all_are_known() -> None:
    candidates = pd.DataFrame(
        {
            "anime_id": pd.Series(
                [10, 20],
                dtype="int64",
            ),
            "score": pd.Series(
                [0.9, 0.8],
                dtype="float64",
            ),
        }
    )

    result = exclude_known_candidates(
        candidates,
        known_anime_ids={10, 20},
    )

    expected = candidates.iloc[0:0].copy().reset_index(drop=True)

    assert_frame_equal(
        result,
        expected,
    )


def test_exclude_known_candidates_rejects_missing_anime_id_column() -> None:
    candidates = pd.DataFrame(
        {
            "score": [0.9],
        }
    )

    with pytest.raises(
        ValueError,
        match="отсутствует столбец anime_id",
    ):
        exclude_known_candidates(
            candidates,
            known_anime_ids=set(),
        )


@pytest.mark.parametrize(
    "candidate_count",
    [
        0,
        -1,
    ],
)
def test_exclude_known_candidates_rejects_invalid_candidate_count(
    candidate_count: int,
) -> None:
    candidates = pd.DataFrame(
        {
            "anime_id": [10],
        }
    )

    with pytest.raises(
        ValueError,
        match="candidate_count",
    ):
        exclude_known_candidates(
            candidates,
            known_anime_ids=set(),
            candidate_count=candidate_count,
        )


def test_exclude_known_candidates_returns_independent_copy() -> None:
    candidates = pd.DataFrame(
        {
            "anime_id": [10, 20],
            "score": [0.9, 0.8],
        }
    )

    result = exclude_known_candidates(
        candidates,
        known_anime_ids=set(),
    )

    result.loc[
        0,
        "anime_id",
    ] = 999

    assert (
        candidates.loc[
            0,
            "anime_id",
        ]
        == 10
    )
