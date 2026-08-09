import numpy as np
import pandas as pd

from shiki_recsys.retrievers.common import (
    RetrieverName,
    build_candidate_frame,
    validate_candidate_count,
)


class PopularityRetriever:
    """Ранжирует аниме по числу положительных train-оценок."""

    def __init__(
        self,
        *,
        relevance_threshold: float,
    ) -> None:
        """
        Инициализирует popularity retriever.

        Args:
            relevance_threshold: Минимальная положительная оценка.

        Raises:
            ValueError: Если порог положительной оценки некорректен.
        """

        if not np.isfinite(relevance_threshold) or not 0 < relevance_threshold <= 10:
            raise ValueError(
                "relevance_threshold должен быть конечным числом "
                "в диапазоне от 0 до 10."
            )

        self._relevance_threshold = float(relevance_threshold)
        self._candidates: pd.DataFrame | None = None
        self._supported_anime_ids: frozenset[int] = frozenset()

    @property
    def supported_anime_ids(self) -> frozenset[int]:
        """
        Возвращает идентификаторы поддерживаемых аниме.

        Returns:
            Идентификаторы аниме, вошедших в рейтинг.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """

        self._require_fitted()

        return self._supported_anime_ids

    def fit(
        self,
        train_interactions: pd.DataFrame,
    ) -> "PopularityRetriever":
        """
        Строит рейтинг по явным train-оценкам.

        Args:
            train_interactions: Обучающие взаимодействия
                с rating больше 0.

        Returns:
            Обученный retriever.

        Raises:
            ValueError: Если данные пусты, не содержат обязательных
                столбцов или включают неявные оценки.
        """

        required_columns = {
            "anime_id",
            "rating",
        }

        missing_columns = required_columns.difference(train_interactions.columns)

        if missing_columns:
            raise ValueError(
                f"В train_interactions отсутствуют столбцы: {sorted(missing_columns)}."
            )

        if train_interactions.empty:
            raise ValueError("train_interactions не должен быть пустым.")

        if (train_interactions["rating"] <= 0).any():
            raise ValueError(
                "PopularityRetriever принимает только "
                "явные взаимодействия с rating больше 0."
            )

        interactions = train_interactions.loc[
            :,
            [
                "anime_id",
                "rating",
            ],
        ].copy()

        interactions["is_positive"] = (
            interactions["rating"] >= self._relevance_threshold
        ).astype("int8")

        popularity_statistics = (
            interactions.groupby(
                "anime_id",
                as_index=False,
            )
            .agg(
                positive_ratings=(
                    "is_positive",
                    "sum",
                ),
                total_ratings=(
                    "rating",
                    "size",
                ),
                mean_rating=(
                    "rating",
                    "mean",
                ),
            )
            .sort_values(
                [
                    "positive_ratings",
                    "total_ratings",
                    "mean_rating",
                    "anime_id",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        candidates = build_candidate_frame(
            anime_ids=popularity_statistics["anime_id"].to_numpy(),
            scores=popularity_statistics["positive_ratings"].to_numpy(),
            source=RetrieverName.POPULARITY,
        )

        self._candidates = candidates
        self._supported_anime_ids = frozenset(candidates["anime_id"].tolist())

        return self

    def retrieve(
        self,
        *,
        candidate_count: int | None = None,
        exclude_anime_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        """
        Возвращает глобально ранжированных кандидатов.

        Args:
            candidate_count: Максимальное количество кандидатов.
                Значение None означает возврат полного рейтинга.
            exclude_anime_ids: Идентификаторы аниме,
                исключаемые из выдачи.

        Returns:
            Таблицу идентификаторов, scores, источников
            и позиций кандидатов.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
            ValueError: Если количество кандидатов некорректно.
        """

        self._require_fitted()

        validate_candidate_count(candidate_count)

        assert self._candidates is not None

        candidates = self._candidates

        if exclude_anime_ids:
            candidates = candidates.loc[~candidates["anime_id"].isin(exclude_anime_ids)]

        return build_candidate_frame(
            anime_ids=candidates["anime_id"].to_numpy(),
            scores=candidates["score"].to_numpy(),
            source=RetrieverName.POPULARITY,
            candidate_count=candidate_count,
        )

    def _require_fitted(self) -> None:
        """
        Проверяет состояние retriever.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """

        if self._candidates is None:
            raise RuntimeError("PopularityRetriever ещё не обучен.")

    def score_items(
        self,
        *,
        anime_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Возвращает popularity scores заданных аниме.

        Args:
            anime_ids: Идентификаторы оцениваемых аниме.

        Returns:
            Scores в порядке переданных anime_id.
        """
        self._require_fitted()

        assert self._candidates is not None

        score_by_anime = self._candidates.set_index("anime_id")["score"]

        return score_by_anime.reindex(anime_ids).to_numpy(dtype=np.float64)
