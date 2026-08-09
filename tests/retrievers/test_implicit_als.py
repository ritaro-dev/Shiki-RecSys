import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from shiki_recsys.retrievers.common import RetrieverName
from shiki_recsys.retrievers.implicit_als import (
    ImplicitALSRetriever,
)


def _make_retriever(
    *,
    factors: int = 2,
    regularization: float = 0.1,
    alpha: float = 1.0,
    iterations: int = 2,
    random_state: int = 42,
) -> ImplicitALSRetriever:
    return ImplicitALSRetriever(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        random_state=random_state,
    )


def _build_signed_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": pd.Series(
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                ],
                dtype="int64",
            ),
            "anime_id": pd.Series(
                [
                    10,
                    20,
                    10,
                    30,
                    20,
                    30,
                ],
                dtype="int64",
            ),
            "confidence": pd.Series(
                [
                    2.0,
                    -1.0,
                    1.0,
                    2.0,
                    0.5,
                    -2.0,
                ],
                dtype="float32",
            ),
        }
    )


@pytest.fixture
def fitted_retriever() -> ImplicitALSRetriever:
    return _make_retriever().fit(_build_signed_interactions())


@pytest.mark.parametrize(
    "retriever_factory",
    [
        pytest.param(
            lambda: _make_retriever(
                factors=0,
            ),
            id="factors_zero",
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
                alpha=0,
            ),
            id="alpha_zero",
        ),
        pytest.param(
            lambda: _make_retriever(
                alpha=math.inf,
            ),
            id="alpha_infinite",
        ),
        pytest.param(
            lambda: _make_retriever(
                iterations=0,
            ),
            id="iterations_zero",
        ),
    ],
)
def test_implicit_als_rejects_invalid_parameters(
    retriever_factory: Callable[
        [],
        ImplicitALSRetriever,
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
        )


@pytest.mark.parametrize(
    "missing_column",
    [
        "user_id",
        "anime_id",
        "confidence",
    ],
)
def test_fit_rejects_missing_columns(
    missing_column: str,
) -> None:
    signed_interactions = _build_signed_interactions().drop(columns=missing_column)

    with pytest.raises(
        ValueError,
        match=missing_column,
    ):
        _make_retriever().fit(signed_interactions)


def test_fit_rejects_empty_interactions() -> None:
    signed_interactions = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "anime_id": pd.Series(dtype="int64"),
            "confidence": pd.Series(dtype="float32"),
        }
    )

    with pytest.raises(
        ValueError,
        match="не должен быть пустым",
    ):
        _make_retriever().fit(signed_interactions)


@pytest.mark.parametrize(
    "invalid_confidence",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_fit_rejects_non_finite_confidence(
    invalid_confidence: float,
) -> None:
    signed_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "confidence": [invalid_confidence],
        }
    )

    with pytest.raises(
        ValueError,
        match="NaN|бесконечные",
    ):
        _make_retriever().fit(signed_interactions)


def test_fit_rejects_zero_confidence() -> None:
    signed_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "confidence": [0.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="ненулевые confidence",
    ):
        _make_retriever().fit(signed_interactions)


def test_fit_rejects_only_negative_signals() -> None:
    signed_interactions = pd.DataFrame(
        {
            "user_id": [1, 2],
            "anime_id": [10, 20],
            "confidence": [-1.0, -2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="не содержит положительных сигналов",
    ):
        _make_retriever().fit(signed_interactions)


def test_fit_sets_supported_anime_ids(
    fitted_retriever: ImplicitALSRetriever,
) -> None:
    assert fitted_retriever.supported_anime_ids == frozenset(
        {
            10,
            20,
            30,
        }
    )


def test_retrieve_returns_ranked_candidates_for_known_user(
    fitted_retriever: ImplicitALSRetriever,
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
        30,
    }

    assert candidates["score"].is_monotonic_decreasing
    assert np.isfinite(candidates["score"]).all()

    assert candidates["source"].tolist() == [
        RetrieverName.IMPLICIT_ALS.value,
        RetrieverName.IMPLICIT_ALS.value,
        RetrieverName.IMPLICIT_ALS.value,
    ]

    assert candidates["source_rank"].tolist() == [
        1,
        2,
        3,
    ]

    assert candidates["anime_id"].dtype == "int64"
    assert candidates["score"].dtype == "float64"
    assert candidates["source"].dtype == "string"
    assert candidates["source_rank"].dtype == "int32"


def test_retrieve_limits_candidate_count(
    fitted_retriever: ImplicitALSRetriever,
) -> None:
    candidates = fitted_retriever.retrieve(
        user_id=1,
        candidate_count=2,
    )

    assert len(candidates) == 2

    assert candidates["source_rank"].tolist() == [
        1,
        2,
    ]


@pytest.mark.parametrize(
    "candidate_count",
    [
        0,
        -1,
    ],
)
def test_retrieve_rejects_invalid_candidate_count(
    fitted_retriever: ImplicitALSRetriever,
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
    fitted_retriever: ImplicitALSRetriever,
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
    fitted_retriever: ImplicitALSRetriever,
) -> None:
    """Проверяет исключение аниме до ограничения выдачи."""

    full_candidates = fitted_retriever.retrieve(
        user_id=1,
    )
    excluded_anime_id = int(full_candidates.iloc[0]["anime_id"])

    candidates = fitted_retriever.retrieve(
        user_id=1,
        candidate_count=2,
        exclude_anime_ids={excluded_anime_id},
    )

    assert candidates["anime_id"].tolist() == (
        full_candidates.iloc[1:3]["anime_id"].tolist()
    )
    assert candidates["source_rank"].tolist() == [1, 2]


def test_score_items_returns_scores_in_requested_order(
    fitted_retriever: ImplicitALSRetriever,
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


def test_supports_user_rejects_call_before_fit() -> None:
    """Проверяет запрет проверки пользователя до обучения."""
    retriever = _make_retriever()

    with pytest.raises(
        RuntimeError,
        match="ещё не обучен",
    ):
        retriever.supports_user(1)


def test_supports_user_checks_trained_user(
    fitted_retriever: ImplicitALSRetriever,
) -> None:
    """Проверяет поддержку известного и неизвестного пользователя."""
    assert fitted_retriever.supports_user(1)
    assert not fitted_retriever.supports_user(999)
