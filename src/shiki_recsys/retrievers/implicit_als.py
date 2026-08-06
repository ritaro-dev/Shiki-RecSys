import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


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
    ) -> pd.DataFrame:
        """
        Возвращает ранжированных кандидатов пользователя.

        Args:
            user_id: Идентификатор пользователя.
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

        inner_user_id = self._user_to_inner.get(user_id)

        if inner_user_id is None:
            return self._empty_candidates()

        assert self._model is not None

        scores = self._model.item_factors @ self._model.user_factors[inner_user_id]

        candidates = (
            pd.DataFrame(
                {
                    "anime_id": self._raw_anime_ids,
                    "score": scores.astype("float64"),
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

        candidates["source"] = pd.Series(
            "implicit_als",
            index=candidates.index,
            dtype="string",
        )

        candidates["source_rank"] = np.arange(
            1,
            len(candidates) + 1,
            dtype=np.int32,
        )

        candidates = candidates[
            [
                "anime_id",
                "score",
                "source",
                "source_rank",
            ]
        ]

        if candidate_count is not None:
            candidates = candidates.head(candidate_count)

        return candidates.copy().reset_index(drop=True)

    def _require_fitted(self) -> None:
        """
        Проверяет состояние retriever.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """
        if self._model is None:
            raise RuntimeError("ImplicitALSRetriever ещё не обучен.")

    @staticmethod
    def _empty_candidates() -> pd.DataFrame:
        """
        Создаёт пустую таблицу кандидатов.

        Returns:
            Пустую таблицу стандартного формата.
        """
        return pd.DataFrame(
            {
                "anime_id": pd.Series(dtype="int64"),
                "score": pd.Series(dtype="float64"),
                "source": pd.Series(dtype="string"),
                "source_rank": pd.Series(dtype="int32"),
            }
        )
