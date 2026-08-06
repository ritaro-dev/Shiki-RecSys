from collections.abc import Collection

import pandas as pd


def exclude_known_candidates(
    candidates: pd.DataFrame,
    *,
    known_anime_ids: Collection[int],
    candidate_count: int | None = None,
) -> pd.DataFrame:
    """
    Исключает из ранжированного списка известные пользователю аниме.

    Args:
        candidates: Ранжированная таблица кандидатов.
        known_anime_ids: Идентификаторы известных пользователю аниме.
        candidate_count: Максимальное количество кандидатов.
            Значение None означает отсутствие ограничения.

    Returns:
        Отфильтрованную таблицу кандидатов с сохранённым порядком.

    Raises:
        ValueError: Если отсутствует столбец anime_id или количество
            кандидатов некорректно.
    """

    if "anime_id" not in candidates.columns:
        raise ValueError("В candidates отсутствует столбец anime_id.")

    if candidate_count is not None and candidate_count <= 0:
        raise ValueError("candidate_count должен быть больше 0 или равен None.")

    filtered_candidates = candidates.loc[
        ~candidates["anime_id"].isin(known_anime_ids)
    ].copy()

    if candidate_count is not None:
        filtered_candidates = filtered_candidates.head(candidate_count)

    return filtered_candidates.reset_index(drop=True)
