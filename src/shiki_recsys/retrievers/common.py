from enum import StrEnum

import numpy as np
import pandas as pd


class RetrieverName(StrEnum):
    """Определяет имена retriever-ов системы."""

    POPULARITY = "popularity"
    EXPLICIT_SVD = "explicit_svd"
    IMPLICIT_ALS = "implicit_als"
    CONTENT_TFIDF = "content_tfidf"


CANDIDATE_COLUMNS = (
    "anime_id",
    "score",
    "source",
    "source_rank",
)


def empty_candidates() -> pd.DataFrame:
    """
    Создаёт пустую таблицу кандидатов стандартного формата.

    Returns:
        Пустую таблицу кандидатов с каноническими типами столбцов.
    """
    return pd.DataFrame(
        {
            "anime_id": pd.Series(dtype="int64"),
            "score": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="string"),
            "source_rank": pd.Series(dtype="int32"),
        }
    )


def validate_candidate_count(
    candidate_count: int | None,
) -> None:
    """
    Проверяет ограничение количества кандидатов.

    Args:
        candidate_count: Максимальное количество кандидатов
            или None.

    Raises:
        ValueError: Если candidate_count не больше 0.
    """
    if candidate_count is not None and candidate_count <= 0:
        raise ValueError("candidate_count должен быть больше 0 или равен None.")


def build_candidate_frame(
    anime_ids: np.ndarray,
    scores: np.ndarray,
    *,
    source: RetrieverName,
    candidate_count: int | None = None,
) -> pd.DataFrame:
    """
    Формирует стандартную таблицу ранжированных кандидатов.

    Args:
        anime_ids: Идентификаторы аниме в порядке ранжирования.
        scores: Scores аниме в том же порядке.
        source: Источник кандидатов.
        candidate_count: Максимальное количество кандидатов
            или None.

    Returns:
        Таблицу кандидатов стандартного формата.

    Raises:
        ValueError: Если candidate_count некорректен.
    """

    candidates = pd.DataFrame(
        {
            "anime_id": pd.Series(anime_ids, dtype="int64"),
            "score": pd.Series(scores, dtype="float64"),
            "source": pd.Series(
                source.value,
                index=range(len(anime_ids)),
                dtype="string",
            ),
            "source_rank": np.arange(
                1,
                len(anime_ids) + 1,
                dtype=np.int32,
            ),
        }
    ).loc[:, CANDIDATE_COLUMNS]

    if candidate_count is not None:
        candidates = candidates.head(candidate_count)

    return candidates.copy().reset_index(drop=True)
