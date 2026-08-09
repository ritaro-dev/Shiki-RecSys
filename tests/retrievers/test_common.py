import numpy as np
import pytest

from shiki_recsys.retrievers.common import (
    CANDIDATE_COLUMNS,
    RetrieverName,
    build_candidate_frame,
    empty_candidates,
    validate_candidate_count,
)


def test_empty_candidates() -> None:
    """Проверяет стандартную пустую таблицу кандидатов."""

    candidates = empty_candidates()

    assert candidates.empty
    assert candidates.columns.tolist() == list(CANDIDATE_COLUMNS)

    assert candidates["anime_id"].dtype == "int64"
    assert candidates["score"].dtype == "float64"
    assert candidates["source"].dtype == "string"
    assert candidates["source_rank"].dtype == "int32"


@pytest.mark.parametrize(
    "candidate_count",
    [0, -1],
)
def test_validate_candidate_count_rejects_invalid_value(
    candidate_count: int,
) -> None:
    """Проверяет отклонение некорректного ограничения кандидатов."""

    with pytest.raises(ValueError):
        validate_candidate_count(candidate_count)


@pytest.mark.parametrize(
    "candidate_count",
    [None, 1, 100],
)
def test_validate_candidate_count_accepts_valid_value(
    candidate_count: int | None,
) -> None:
    """Проверяет допустимые ограничения количества кандидатов."""

    validate_candidate_count(candidate_count)


def test_build_candidate_frame() -> None:
    """Проверяет формирование стандартной таблицы кандидатов."""

    candidates = build_candidate_frame(
        anime_ids=np.array([30, 10, 20]),
        scores=np.array([0.9, 0.8, 0.7]),
        source=RetrieverName.CONTENT_TFIDF,
        candidate_count=2,
    )

    assert candidates.columns.tolist() == list(CANDIDATE_COLUMNS)
    assert candidates["anime_id"].tolist() == [30, 10]
    assert candidates["score"].tolist() == [0.9, 0.8]
    assert candidates["source"].tolist() == [
        RetrieverName.CONTENT_TFIDF.value,
        RetrieverName.CONTENT_TFIDF.value,
    ]
    assert candidates["source_rank"].tolist() == [1, 2]

    assert candidates["anime_id"].dtype == "int64"
    assert candidates["score"].dtype == "float64"
    assert candidates["source"].dtype == "string"
    assert candidates["source_rank"].dtype == "int32"
