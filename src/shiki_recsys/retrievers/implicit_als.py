import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from shiki_recsys.retrievers.common import (
    RetrieverName,
    build_candidate_frame,
    empty_candidates,
    exclude_scored_items,
    validate_candidate_count,
)


class ImplicitALSRetriever:
    """Формирует персональные кандидаты с помощью implicit ALS."""

    def __init__(
        self,
        *,
        factors: int,
        regularization: float,
        alpha: float,
        iterations: int,
        random_state: int,
    ) -> None:
        """
        Инициализирует implicit ALS retriever.

        Args:
            factors: Размерность скрытых факторов.
            regularization: Коэффициент регуляризации.
            alpha: Общий множитель confidence.
            iterations: Количество итераций обучения.
            random_state: Начальное значение генератора
                случайных чисел.

        Raises:
            ValueError: Если параметры retriever некорректны.
        """
        if factors <= 0:
            raise ValueError("factors должен быть больше 0.")

        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization должен быть конечным числом не меньше 0.")

        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha должен быть конечным числом больше 0.")

        if iterations <= 0:
            raise ValueError("iterations должен быть больше 0.")

        self._factors = factors
        self._regularization = regularization
        self._alpha = alpha
        self._iterations = iterations
        self._random_state = random_state

        self._model: AlternatingLeastSquares | None = None
        self._user_to_inner: dict[int, int] = {}
        self._anime_to_inner: dict[int, int] = {}
        self._raw_anime_ids = np.array(
            [],
            dtype=np.int64,
        )
        self._supported_anime_ids: frozenset[int] = frozenset()

    @property
    def supported_anime_ids(self) -> frozenset[int]:
        """
        Возвращает идентификаторы поддерживаемых аниме.

        Returns:
            Идентификаторы аниме, использованные при обучении.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """
        self._require_fitted()
        return self._supported_anime_ids

    def fit(
        self,
        signed_interactions: pd.DataFrame,
    ) -> "ImplicitALSRetriever":
        """
        Обучает ALS на signed-взаимодействиях.

        Args:
            signed_interactions: Взаимодействия со столбцами
                user_id, anime_id и ненулевым confidence.

        Returns:
            Обученный retriever.

        Raises:
            ValueError: Если обучающие данные некорректны
                или не содержат положительных сигналов.
        """
        required_columns = {
            "user_id",
            "anime_id",
            "confidence",
        }
        missing_columns = required_columns.difference(signed_interactions.columns)

        if missing_columns:
            raise ValueError(
                f"В signed_interactions отсутствуют столбцы: {sorted(missing_columns)}."
            )

        if signed_interactions.empty:
            raise ValueError("signed_interactions не должен быть пустым.")

        if not np.isfinite(signed_interactions["confidence"]).all():
            raise ValueError(
                "Столбец confidence содержит NaN или бесконечные значения."
            )

        if (signed_interactions["confidence"] == 0).any():
            raise ValueError(
                "signed_interactions должен содержать только ненулевые confidence."
            )

        if not (signed_interactions["confidence"] > 0).any():
            raise ValueError("signed_interactions не содержит положительных сигналов.")

        raw_user_ids = np.array(
            sorted(signed_interactions["user_id"].unique()),
            dtype=np.int64,
        )

        raw_anime_ids = np.array(
            sorted(signed_interactions["anime_id"].unique()),
            dtype=np.int64,
        )

        user_to_inner = {
            int(user_id): int(inner_user_id)
            for inner_user_id, user_id in enumerate(raw_user_ids)
        }

        anime_to_inner = {
            int(anime_id): int(inner_anime_id)
            for inner_anime_id, anime_id in enumerate(raw_anime_ids)
        }

        row_indices = (
            signed_interactions["user_id"].map(user_to_inner).to_numpy(dtype=np.int32)
        )

        column_indices = (
            signed_interactions["anime_id"].map(anime_to_inner).to_numpy(dtype=np.int32)
        )

        confidence_values = signed_interactions["confidence"].to_numpy(dtype=np.float32)

        user_item_matrix = csr_matrix(
            (
                confidence_values,
                (
                    row_indices,
                    column_indices,
                ),
            ),
            shape=(
                len(raw_user_ids),
                len(raw_anime_ids),
            ),
            dtype=np.float32,
        )

        user_item_matrix.sum_duplicates()
        user_item_matrix.eliminate_zeros()

        if user_item_matrix.nnz == 0:
            raise ValueError("Построенная ALS-матрица не содержит взаимодействий.")

        if not (user_item_matrix.data > 0).any():
            raise ValueError(
                "Построенная ALS-матрица не содержит положительных сигналов."
            )

        model = AlternatingLeastSquares(
            factors=self._factors,
            regularization=self._regularization,
            alpha=self._alpha,
            iterations=self._iterations,
            calculate_training_loss=False,
            random_state=self._random_state,
        )

        model.fit(
            user_item_matrix,
            show_progress=False,
        )

        self._model = model
        self._user_to_inner = user_to_inner
        self._anime_to_inner = anime_to_inner
        self._raw_anime_ids = raw_anime_ids
        self._supported_anime_ids = frozenset(
            int(anime_id) for anime_id in raw_anime_ids
        )

        return self

    def retrieve(
        self,
        *,
        user_id: int,
        candidate_count: int | None = None,
        exclude_anime_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        """
        Возвращает ранжированных кандидатов пользователя.

        Args:
            user_id: Идентификатор пользователя.
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

        inner_user_id = self._user_to_inner.get(user_id)

        if inner_user_id is None:
            return empty_candidates()

        assert self._model is not None

        scores = self._model.item_factors @ self._model.user_factors[inner_user_id]

        anime_ids = self._raw_anime_ids

        anime_ids, scores = exclude_scored_items(
            anime_ids,
            scores,
            exclude_anime_ids,
        )

        ranked_items = (
            pd.DataFrame(
                {
                    "anime_id": anime_ids,
                    "score": scores,
                }
            )
            .sort_values(
                [
                    "score",
                    "anime_id",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        return build_candidate_frame(
            anime_ids=ranked_items["anime_id"].to_numpy(),
            scores=ranked_items["score"].to_numpy(),
            source=RetrieverName.IMPLICIT_ALS,
            candidate_count=candidate_count,
        )

    def _require_fitted(self) -> None:
        """
        Проверяет состояние retriever.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """
        if self._model is None:
            raise RuntimeError("ImplicitALSRetriever ещё не обучен.")

    def score_items(
        self,
        *,
        user_id: int,
        anime_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Рассчитывает ALS scores для заданных аниме.

        Args:
            user_id: Идентификатор пользователя.
            anime_ids: Идентификаторы оцениваемых аниме.

        Returns:
            Scores в порядке переданных anime_id.
        """
        self._require_fitted()

        scores = np.full(
            len(anime_ids),
            np.nan,
            dtype=np.float64,
        )

        inner_user_id = self._user_to_inner.get(user_id)

        if inner_user_id is None:
            return scores

        assert self._model is not None

        supported_positions = [
            position
            for position, anime_id in enumerate(anime_ids)
            if int(anime_id) in self._anime_to_inner
        ]

        if not supported_positions:
            return scores

        inner_anime_ids = np.array(
            [
                self._anime_to_inner[int(anime_ids[position])]
                for position in supported_positions
            ],
            dtype=np.int64,
        )

        scores[supported_positions] = (
            self._model.item_factors[inner_anime_ids]
            @ self._model.user_factors[inner_user_id]
        )

        return scores
