from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from shiki_recsys.inference import warm_pipeline


def test_build_warm_ranking_sorts_ranker_scores(monkeypatch) -> None:
    """Проверяет построение warm ranking по score ranker-а."""

    features = pd.DataFrame(
        {
            "user_id": [7, 7, 7],
            "anime_id": [30, 10, 20],
            "score_popularity": [1.0, 2.0, 3.0],
        }
    )

    build_features = MagicMock(return_value=features)
    monkeypatch.setattr(
        warm_pipeline,
        "build_ranker_features_for_user",
        build_features,
    )

    popularity = MagicMock()
    explicit_svd = MagicMock()
    implicit_als = MagicMock()
    content_tfidf = MagicMock()
    ranker = MagicMock()
    ranker.predict.return_value = np.array(
        [0.2, 0.9, 0.9],
        dtype=np.float64,
    )

    ranking = warm_pipeline.build_warm_ranking(
        user_id=7,
        known_anime_ids={100, 200},
        candidate_count=100,
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
        ranker=ranker,
    )

    assert ranking["anime_id"].tolist() == [10, 20, 30]
    assert ranking["ranker_score"].tolist() == [0.9, 0.9, 0.2]
    assert ranking["rank"].tolist() == [1, 2, 3]

    build_features.assert_called_once_with(
        user_id=7,
        known_anime_ids={100, 200},
        candidate_count=100,
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
    )
    ranker.predict.assert_called_once_with(features)


def test_build_warm_ranking_handles_empty_candidates(monkeypatch) -> None:
    """Проверяет пустой warm ranking без вызова ranker-а."""

    monkeypatch.setattr(
        warm_pipeline,
        "build_ranker_features_for_user",
        MagicMock(return_value=pd.DataFrame()),
    )

    ranker = MagicMock()

    ranking = warm_pipeline.build_warm_ranking(
        user_id=7,
        known_anime_ids=set(),
        candidate_count=100,
        popularity=MagicMock(),
        explicit_svd=MagicMock(),
        implicit_als=MagicMock(),
        content_tfidf=MagicMock(),
        ranker=ranker,
    )

    assert ranking.empty
    assert ranking.columns.tolist() == [
        "anime_id",
        "ranker_score",
        "rank",
    ]
    assert ranking["anime_id"].dtype == "int64"
    assert ranking["ranker_score"].dtype == "float64"
    assert ranking["rank"].dtype == "int32"

    ranker.predict.assert_not_called()
