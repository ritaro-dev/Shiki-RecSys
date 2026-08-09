import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from shiki_recsys.retrievers.common import RetrieverName
from shiki_recsys.retrievers.popularity import (
    PopularityRetriever,
)


@pytest.mark.parametrize(
    "relevance_threshold",
    [
        0,
        -1,
        10.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_init_rejects_invalid_relevance_threshold(
    relevance_threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="relevance_threshold",
    ):
        PopularityRetriever(
            relevance_threshold=relevance_threshold,
        )


def test_retriever_rejects_access_before_fit() -> None:
    retriever = PopularityRetriever(
        relevance_threshold=8,
    )

    with pytest.raises(
        RuntimeError,
        match="ещё не обучен",
    ):
        _ = retriever.supported_anime_ids

    with pytest.raises(
        RuntimeError,
        match="ещё не обучен",
    ):
        retriever.retrieve()


@pytest.mark.parametrize(
    "train_interactions",
    [
        pd.DataFrame(
            {
                "rating": [8.0],
            }
        ),
        pd.DataFrame(
            {
                "anime_id": [1],
            }
        ),
    ],
)
def test_fit_rejects_missing_columns(
    train_interactions: pd.DataFrame,
) -> None:
    retriever = PopularityRetriever(
        relevance_threshold=8,
    )

    with pytest.raises(
        ValueError,
        match="отсутствуют столбцы",
    ):
        retriever.fit(train_interactions)


def test_fit_rejects_empty_interactions() -> None:
    interactions = pd.DataFrame(
        {
            "anime_id": pd.Series(
                dtype="int64",
            ),
            "rating": pd.Series(
                dtype="float32",
            ),
        }
    )

    retriever = PopularityRetriever(
        relevance_threshold=8,
    )

    with pytest.raises(
        ValueError,
        match="не должен быть пустым",
    ):
        retriever.fit(interactions)


@pytest.mark.parametrize(
    "invalid_rating",
    [
        0,
        -1,
    ],
)
def test_fit_rejects_non_explicit_interactions(
    invalid_rating: float,
) -> None:
    interactions = pd.DataFrame(
        {
            "anime_id": [
                1,
                2,
            ],
            "rating": [
                8,
                invalid_rating,
            ],
        }
    )

    retriever = PopularityRetriever(
        relevance_threshold=8,
    )

    with pytest.raises(
        ValueError,
        match="rating больше 0",
    ):
        retriever.fit(interactions)


def test_fit_builds_expected_popularity_ranking() -> None:
    interactions = pd.DataFrame(
        {
            "anime_id": [
                10,
                10,
                10,
                20,
                20,
                20,
                20,
                30,
                30,
                30,
                30,
                40,
                40,
                50,
                50,
            ],
            "rating": [
                10,
                9,
                7,
                8,
                8,
                7,
                6,
                9,
                8,
                7,
                7,
                8,
                7,
                8,
                7,
            ],
        }
    )

    retriever = PopularityRetriever(
        relevance_threshold=8,
    )

    result = retriever.fit(interactions).retrieve()

    expected = pd.DataFrame(
        {
            "anime_id": pd.Series(
                [
                    30,
                    20,
                    10,
                    40,
                    50,
                ],
                dtype="int64",
            ),
            "score": pd.Series(
                [
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                ],
                dtype="float64",
            ),
            "source": pd.Series(
                [
                    RetrieverName.POPULARITY.value,
                    RetrieverName.POPULARITY.value,
                    RetrieverName.POPULARITY.value,
                    RetrieverName.POPULARITY.value,
                    RetrieverName.POPULARITY.value,
                ],
                dtype="string",
            ),
            "source_rank": pd.Series(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],
                dtype="int32",
            ),
        }
    )

    assert_frame_equal(
        result,
        expected,
    )

    assert retriever.supported_anime_ids == frozenset(
        {
            10,
            20,
            30,
            40,
            50,
        }
    )


def test_retrieve_limits_candidate_count() -> None:
    interactions = pd.DataFrame(
        {
            "anime_id": [
                1,
                2,
                3,
            ],
            "rating": [
                10,
                9,
                8,
            ],
        }
    )

    retriever = PopularityRetriever(
        relevance_threshold=8,
    ).fit(interactions)

    full_result = retriever.retrieve()

    limited_result = retriever.retrieve(
        candidate_count=2,
    )

    expected = full_result.head(2).reset_index(drop=True)

    assert_frame_equal(
        limited_result,
        expected,
    )


@pytest.mark.parametrize(
    "candidate_count",
    [
        0,
        -1,
    ],
)
def test_retrieve_rejects_invalid_candidate_count(
    candidate_count: int,
) -> None:
    interactions = pd.DataFrame(
        {
            "anime_id": [1],
            "rating": [10],
        }
    )

    retriever = PopularityRetriever(
        relevance_threshold=8,
    ).fit(interactions)

    with pytest.raises(
        ValueError,
        match="candidate_count",
    ):
        retriever.retrieve(
            candidate_count=candidate_count,
        )


def test_retrieve_returns_independent_copy() -> None:
    interactions = pd.DataFrame(
        {
            "anime_id": [
                1,
                2,
            ],
            "rating": [
                10,
                9,
            ],
        }
    )

    retriever = PopularityRetriever(
        relevance_threshold=8,
    ).fit(interactions)

    first_result = retriever.retrieve()

    first_result.loc[
        0,
        "anime_id",
    ] = 999

    second_result = retriever.retrieve()

    assert (
        second_result.loc[
            0,
            "anime_id",
        ]
        == 1
    )


def test_retrieve_excludes_anime_before_candidate_limit() -> None:
    """Проверяет исключение аниме до ограничения выдачи."""

    interactions = pd.DataFrame(
        {
            "anime_id": [1, 2, 3],
            "rating": [10, 9, 8],
        }
    )

    retriever = PopularityRetriever(
        relevance_threshold=8,
    ).fit(interactions)

    candidates = retriever.retrieve(
        candidate_count=2,
        exclude_anime_ids={1},
    )

    assert candidates["anime_id"].tolist() == [2, 3]
    assert candidates["source_rank"].tolist() == [1, 2]


def test_score_items_returns_scores_in_requested_order() -> None:
    """Проверяет scores заданных аниме и неподдерживаемые объекты."""

    interactions = pd.DataFrame(
        {
            "anime_id": [10, 10, 20, 30],
            "rating": [10, 9, 8, 7],
        }
    )

    retriever = PopularityRetriever(
        relevance_threshold=8,
    ).fit(interactions)

    scores = retriever.score_items(
        anime_ids=np.array(
            [20, 999, 10],
            dtype=np.int64,
        )
    )

    np.testing.assert_allclose(
        scores[[0, 2]],
        [1.0, 2.0],
    )
    assert np.isnan(scores[1])
