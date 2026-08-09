import numpy as np
import pandas as pd
import pytest

from shiki_recsys.config.training import RankerConfig
from shiki_recsys.ranking.catboost import CatBoostRankerModel


def test_catboost_ranker_fits_and_predicts() -> None:
    """Проверяет обучение и получение ranking scores."""

    data = pd.DataFrame(
        {
            "user_id": [2, 1, 2, 1, 2, 1],
            "anime_id": [20, 10, 21, 11, 22, 12],
            "score_popularity": [3.0, 5.0, 2.0, 4.0, 1.0, 3.0],
            "rank_popularity": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "retriever_count": [2, 3, 1, 2, 1, 1],
            "target": [1, 1, 0, 0, 0, 0],
        }
    )

    ranker = CatBoostRankerModel(
        RankerConfig(
            iterations=5,
            depth=2,
            learning_rate=0.1,
            l2_leaf_reg=1.0,
        ),
        random_seed=42,
    )

    ranker.fit(data)

    predictions = ranker.predict(data.drop(columns="target"))

    assert predictions.shape == (len(data),)
    assert np.isfinite(predictions).all()


def test_catboost_ranker_rejects_prediction_before_fit() -> None:
    """Проверяет запрет inference до обучения."""

    ranker = CatBoostRankerModel(
        RankerConfig(
            iterations=5,
            depth=2,
            learning_rate=0.1,
            l2_leaf_reg=1.0,
        ),
        random_seed=42,
    )

    data = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "score_popularity": [1.0],
        }
    )

    with pytest.raises(
        RuntimeError,
        match="Ranker is not fitted",
    ):
        ranker.predict(data)
