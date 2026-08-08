import math

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.trainset import Trainset

from shiki_recsys.retrievers.common import (
    RetrieverName,
    build_candidate_frame,
    empty_candidates,
    validate_candidate_count,
)


class ExplicitSVDRetriever:
    """Формирует персональные кандидаты по явным оценкам."""

    def __init__(
        self,
        *,
        min_item_explicit_ratings: int,
        n_factors: int,
        n_epochs: int,
        biased: bool,
        learning_rate: float,
        regularization: float,
        init_mean: float,
        init_std_dev: float,
        random_state: int,
    ) -> None:
        """
        Инициализирует explicit SVD retriever.

        Args:
            min_item_explicit_ratings: Минимальное количество явных
                train-оценок аниме.
            n_factors: Размерность скрытых факторов.
            n_epochs: Количество эпох обучения.
            biased: Использование глобального среднего
                и смещений пользователей и аниме.
            learning_rate: Скорость обучения.
            regularization: Коэффициент регуляризации.
            init_mean: Среднее начального распределения факторов.
            init_std_dev: Стандартное отклонение начального
                распределения факторов.
            random_state: Начальное значение генератора случайных чисел.

        Raises:
            ValueError: Если параметры retriever некорректны.
        """
        if min_item_explicit_ratings <= 0:
            raise ValueError("min_item_explicit_ratings должен быть больше 0.")
        if n_factors <= 0:
            raise ValueError("n_factors должен быть больше 0.")
        if n_epochs <= 0:
            raise ValueError("n_epochs должен быть больше 0.")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate должен быть конечным числом больше 0.")
        if not math.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization должен быть конечным числом не меньше 0.")
        if not math.isfinite(init_mean):
            raise ValueError("init_mean должен быть конечным числом.")
        if not math.isfinite(init_std_dev) or init_std_dev <= 0:
            raise ValueError("init_std_dev должен быть конечным числом больше 0.")

        self._min_item_explicit_ratings = min_item_explicit_ratings
        self._n_factors = n_factors
        self._n_epochs = n_epochs
        self._biased = biased
        self._learning_rate = learning_rate
        self._regularization = regularization
        self._init_mean = init_mean
        self._init_std_dev = init_std_dev
        self._random_state = random_state

        self._model: SVD | None = None
        self._trainset: Trainset | None = None
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
        train_interactions: pd.DataFrame,
    ) -> "ExplicitSVDRetriever":
        """
        Обучает SVD на явных train-оценках.

        Args:
            train_interactions: Обучающие явные взаимодействия.

        Returns:
            Обученный retriever.

        Raises:
            ValueError: Если обучающие данные некорректны
                или не содержат поддерживаемых аниме.
        """
        required_columns = {
            "user_id",
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

        if (
            not train_interactions["rating"]
            .between(
                1,
                10,
            )
            .all()
        ):
            raise ValueError(
                "ExplicitSVDRetriever принимает только оценки в диапазоне от 1 до 10."
            )

        item_rating_counts = train_interactions.groupby(
            "anime_id",
            sort=False,
        ).size()

        supported_anime_ids = item_rating_counts.loc[
            item_rating_counts >= self._min_item_explicit_ratings
        ].index

        model_interactions = (
            train_interactions.loc[
                train_interactions["anime_id"].isin(supported_anime_ids),
                [
                    "user_id",
                    "anime_id",
                    "rating",
                ],
            ]
            .copy()
            .reset_index(drop=True)
        )

        if model_interactions.empty:
            raise ValueError(
                "После применения min_item_explicit_ratings "
                "не осталось взаимодействий для обучения."
            )

        reader = Reader(
            rating_scale=(1, 10),
        )
        dataset = Dataset.load_from_df(
            model_interactions[
                [
                    "user_id",
                    "anime_id",
                    "rating",
                ]
            ],
            reader,
        )
        trainset = dataset.build_full_trainset()

        model = SVD(
            n_factors=self._n_factors,
            n_epochs=self._n_epochs,
            biased=self._biased,
            lr_all=self._learning_rate,
            reg_all=self._regularization,
            init_mean=self._init_mean,
            init_std_dev=self._init_std_dev,
            random_state=self._random_state,
            verbose=False,
        )
        model.fit(trainset)

        self._model = model
        self._trainset = trainset
        self._supported_anime_ids = frozenset(
            int(anime_id) for anime_id in supported_anime_ids
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

        validate_candidate_count(candidate_count)

        assert self._model is not None
        assert self._trainset is not None

        try:
            inner_user_id = self._trainset.to_inner_uid(user_id)
        except ValueError:
            return empty_candidates()

        item_inner_ids = np.arange(
            self._trainset.n_items,
            dtype=np.int64,
        )

        scores = self._model.qi @ self._model.pu[inner_user_id]

        if self._biased:
            scores = (
                scores
                + self._trainset.global_mean
                + self._model.bu[inner_user_id]
                + self._model.bi
            )

        anime_ids = np.fromiter(
            (
                int(self._trainset.to_raw_iid(int(inner_item_id)))
                for inner_item_id in item_inner_ids
            ),
            dtype=np.int64,
            count=self._trainset.n_items,
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
            source=RetrieverName.EXPLICIT_SVD,
            candidate_count=candidate_count,
        )

    def _require_fitted(self) -> None:
        """
        Проверяет состояние retriever.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """
        if self._model is None or self._trainset is None:
            raise RuntimeError("ExplicitSVDRetriever ещё не обучен.")
