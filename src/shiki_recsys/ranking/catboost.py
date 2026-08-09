import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from shiki_recsys.config.training import RankerConfig


class CatBoostRankerModel:
    """Обучает CatBoost-модель для ранжирования кандидатов."""

    def __init__(
        self,
        config: RankerConfig,
        *,
        random_seed: int,
    ) -> None:
        """
        Инициализирует ranker.

        Args:
            config: Параметры CatBoost ranker.
            random_seed: Seed для воспроизводимости.
        """
        self._model = CatBoostRanker(
            iterations=config.iterations,
            depth=config.depth,
            learning_rate=config.learning_rate,
            l2_leaf_reg=config.l2_leaf_reg,
            loss_function="YetiRank",
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        self._feature_columns: list[str] | None = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Обучает ranker на размеченных кандидатах.

        Args:
            data: Кандидаты с признаками и target.
        """
        train = data.sort_values(
            "user_id",
            kind="stable",
        )

        self._feature_columns = [
            column
            for column in train.columns
            if column not in {"user_id", "anime_id", "target"}
        ]

        self._model.fit(
            train[self._feature_columns],
            train["target"],
            group_id=train["user_id"].to_numpy(),
        )

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Рассчитывает ranking scores кандидатов.

        Args:
            data: Кандидаты с признаками.

        Returns:
            Ranking scores кандидатов.

        Raises:
            RuntimeError: Если модель не обучена.
        """
        if self._feature_columns is None:
            raise RuntimeError("Ranker is not fitted.")

        return self._model.predict(data[self._feature_columns])
