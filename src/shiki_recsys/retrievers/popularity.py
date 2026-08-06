import numpy as np
import pandas as pd


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

        candidates = pd.DataFrame(
            {
                "anime_id": (popularity_statistics["anime_id"].astype("int64")),
                "score": (popularity_statistics["positive_ratings"].astype("float64")),
                "source": pd.Series(
                    "popularity",
                    index=popularity_statistics.index,
                    dtype="string",
                ),
                "source_rank": np.arange(
                    1,
                    len(popularity_statistics) + 1,
                    dtype=np.int32,
                ),
            }
        )

        self._candidates = candidates

        self._supported_anime_ids = frozenset(candidates["anime_id"].tolist())

        return self

    def retrieve(
        self,
        *,
        candidate_count: int | None = None,
    ) -> pd.DataFrame:
        """
        Возвращает глобально ранжированных кандидатов.

        Args:
            candidate_count: Максимальное количество кандидатов.
                Значение None означает возврат полного рейтинга.

        Returns:
            Таблицу идентификаторов, scores, источников
            и позиций кандидатов.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
            ValueError: Если количество кандидатов некорректно.
        """

        self._require_fitted()

        if candidate_count is not None and candidate_count <= 0:
            raise ValueError("candidate_count должен быть больше 0 или равен None.")

        assert self._candidates is not None

        if candidate_count is None:
            return self._candidates.copy()

        return self._candidates.head(candidate_count).copy().reset_index(drop=True)

    def _require_fitted(self) -> None:
        """
        Проверяет состояние retriever.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """

        if self._candidates is None:
            raise RuntimeError("PopularityRetriever ещё не обучен.")
