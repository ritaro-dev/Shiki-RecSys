import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from shiki_recsys.retrievers.common import RetrieverName
from shiki_recsys.retrievers.explicit_svd import (
    ExplicitSVDRetriever,
)


def _make_retriever(
    *,
    min_item_explicit_ratings: int = 2,
    n_factors: int = 2,
    n_epochs: int = 2,
    biased: bool = True,
    learning_rate: float = 0.01,
    regularization: float = 0.1,
    init_mean: float = 0.0,
    init_std_dev: float = 0.1,
    random_state: int = 42,
) -> ExplicitSVDRetriever:
    return ExplicitSVDRetriever(
        min_item_explicit_ratings=min_item_explicit_ratings,
        n_factors=n_factors,
        n_epochs=n_epochs,
        biased=biased,
        learning_rate=learning_rate,
        regularization=regularization,
        init_mean=init_mean,
        init_std_dev=init_std_dev,
        random_state=random_state,
    )


def _build_train_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [
                1,
                1,
                2,
                2,
                3,
                3,
            ],
            "anime_id": [
                10,
                20,
                10,
                20,
                10,
                30,
            ],
            "rating": [
                9.0,
                8.0,
                7.0,
                6.0,
                10.0,
                9.0,
            ],
        }
    )


@pytest.fixture
def fitted_retriever() -> ExplicitSVDRetriever:
    return _make_retriever().fit(_build_train_interactions())


@pytest.mark.parametrize(
    "retriever_factory",
    [
        pytest.param(
            lambda: _make_retriever(
                min_item_explicit_ratings=0,
            ),
            id="min_item_explicit_ratings",
        ),
        pytest.param(
            lambda: _make_retriever(
                n_factors=0,
            ),
            id="n_factors",
        ),
        pytest.param(
            lambda: _make_retriever(
                n_epochs=0,
            ),
            id="n_epochs",
        ),
        pytest.param(
            lambda: _make_retriever(
                learning_rate=0,
            ),
            id="learning_rate_zero",
        ),
        pytest.param(
            lambda: _make_retriever(
                learning_rate=math.inf,
            ),
            id="learning_rate_infinite",
        ),
        pytest.param(
            lambda: _make_retriever(
                regularization=-0.1,
            ),
            id="regularization_negative",
        ),
        pytest.param(
            lambda: _make_retriever(
                regularization=math.nan,
            ),
            id="regularization_nan",
        ),
        pytest.param(
            lambda: _make_retriever(
                init_mean=math.inf,
            ),
            id="init_mean_infinite",
        ),
        pytest.param(
            lambda: _make_retriever(
                init_std_dev=0,
            ),
            id="init_std_dev_zero",
        ),
        pytest.param(
            lambda: _make_retriever(
                init_std_dev=math.nan,
            ),
            id="init_std_dev_nan",
        ),
    ],
)
def test_explicit_svd_rejects_invalid_parameters(
    retriever_factory: Callable[
        [],
        ExplicitSVDRetriever,
    ],
) -> None:
    with pytest.raises(ValueError):
        retriever_factory()


def test_supported_anime_ids_rejects_call_before_fit() -> None:
    retriever = _make_retriever()

    with pytest.raises(
        RuntimeError,
        match="ещё не обучен",
    ):
        _ = retriever.supported_anime_ids


def test_retrieve_rejects_call_before_fit() -> None:
    retriever = _make_retriever()

    with pytest.raises(
        RuntimeError,
        match="ещё не обучен",
    ):
        retriever.retrieve(
            user_id=1,
            candidate_count=10,
        )


def test_fit_rejects_missing_columns() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
        }
    )

    with pytest.raises(
        ValueError,
        match="rating",
    ):
        _make_retriever().fit(interactions)


def test_fit_rejects_empty_interactions() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "anime_id": pd.Series(dtype="int64"),
            "rating": pd.Series(dtype="float32"),
        }
    )

    with pytest.raises(
        ValueError,
        match="не должен быть пустым",
    ):
        _make_retriever().fit(interactions)


@pytest.mark.parametrize(
    "invalid_rating",
    [
        0.0,
        11.0,
        math.nan,
    ],
)
def test_fit_rejects_invalid_explicit_ratings(
    invalid_rating: float,
) -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "rating": [invalid_rating],
        }
    )

    with pytest.raises(
        ValueError,
        match="от 1 до 10",
    ):
        _make_retriever(
            min_item_explicit_ratings=1,
        ).fit(interactions)


def test_fit_rejects_dataset_without_supported_items() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [
                1,
                2,
            ],
            "anime_id": [
                10,
                20,
            ],
            "rating": [
                8.0,
                9.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="не осталось взаимодействий",
    ):
        _make_retriever(
            min_item_explicit_ratings=2,
        ).fit(interactions)


def test_fit_keeps_only_items_with_enough_explicit_ratings(
    fitted_retriever: ExplicitSVDRetriever,
) -> None:
    assert fitted_retriever.supported_anime_ids == frozenset(
        {
            10,
            20,
        }
    )


def test_retrieve_returns_ranked_candidates_for_known_user(
    fitted_retriever: ExplicitSVDRetriever,
) -> None:
    candidates = fitted_retriever.retrieve(
        user_id=1,
    )

    assert candidates.columns.tolist() == [
        "anime_id",
        "score",
        "source",
        "source_rank",
    ]

    assert set(candidates["anime_id"]) == {
        10,
        20,
    }

    assert candidates["score"].is_monotonic_decreasing

    assert candidates["source"].tolist() == [
        RetrieverName.EXPLICIT_SVD.value,
        RetrieverName.EXPLICIT_SVD.value,
    ]

    assert candidates["source_rank"].tolist() == [
        1,
        2,
    ]

    assert candidates["anime_id"].dtype == "int64"
    assert candidates["score"].dtype == "float64"
    assert candidates["source"].dtype == "string"
    assert candidates["source_rank"].dtype == "int32"


def test_retrieve_limits_candidate_count(
    fitted_retriever: ExplicitSVDRetriever,
) -> None:
    candidates = fitted_retriever.retrieve(
        user_id=1,
        candidate_count=1,
    )

    assert len(candidates) == 1
    assert candidates["source_rank"].tolist() == [1]


@pytest.mark.parametrize(
    "candidate_count",
    [
        0,
        -1,
    ],
)
def test_retrieve_rejects_invalid_candidate_count(
    fitted_retriever: ExplicitSVDRetriever,
    candidate_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="candidate_count",
    ):
        fitted_retriever.retrieve(
            user_id=1,
            candidate_count=candidate_count,
        )


def test_retrieve_returns_typed_empty_frame_for_unknown_user(
    fitted_retriever: ExplicitSVDRetriever,
) -> None:
    candidates = fitted_retriever.retrieve(
        user_id=999,
    )

    assert candidates.empty

    assert candidates.columns.tolist() == [
        "anime_id",
        "score",
        "source",
        "source_rank",
    ]

    assert candidates["anime_id"].dtype == "int64"
    assert candidates["score"].dtype == "float64"
    assert candidates["source"].dtype == "string"
    assert candidates["source_rank"].dtype == "int32"


def test_retrieve_excludes_anime_before_candidate_limit(
    fitted_retriever: ExplicitSVDRetriever,
) -> None:
    """Проверяет исключение аниме до ограничения выдачи."""

    full_candidates = fitted_retriever.retrieve(
        user_id=1,
    )
    excluded_anime_id = int(full_candidates.iloc[0]["anime_id"])

    candidates = fitted_retriever.retrieve(
        user_id=1,
        candidate_count=1,
        exclude_anime_ids={excluded_anime_id},
    )

    assert candidates["anime_id"].tolist() == [int(full_candidates.iloc[1]["anime_id"])]
    assert candidates["source_rank"].tolist() == [1]


def test_score_items_returns_scores_in_requested_order(
    fitted_retriever: ExplicitSVDRetriever,
) -> None:
    """Проверяет scores заданных аниме и неподдерживаемые объекты."""

    full_candidates = fitted_retriever.retrieve(user_id=1).set_index("anime_id")[
        "score"
    ]

    anime_ids = np.array(
        [20, 999, 10],
        dtype=np.int64,
    )

    scores = fitted_retriever.score_items(
        user_id=1,
        anime_ids=anime_ids,
    )

    np.testing.assert_allclose(
        scores[[0, 2]],
        [
            full_candidates.loc[20],
            full_candidates.loc[10],
        ],
    )
    assert np.isnan(scores[1])

    unknown_user_scores = fitted_retriever.score_items(
        user_id=999,
        anime_ids=anime_ids,
    )

    assert np.isnan(unknown_user_scores).all()
