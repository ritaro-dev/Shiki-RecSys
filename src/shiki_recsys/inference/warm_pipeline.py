import numpy as np
import pandas as pd

from shiki_recsys.ranking.candidates import build_ranker_features_for_user
from shiki_recsys.ranking.catboost import CatBoostRankerModel
from shiki_recsys.retrievers.content_tfidf import ContentTFIDFRetriever
from shiki_recsys.retrievers.explicit_svd import ExplicitSVDRetriever
from shiki_recsys.retrievers.implicit_als import ImplicitALSRetriever
from shiki_recsys.retrievers.popularity import PopularityRetriever


def build_warm_ranking(
    *,
    user_id: int,
    known_anime_ids: set[int],
    candidate_count: int,
    popularity: PopularityRetriever,
    explicit_svd: ExplicitSVDRetriever,
    implicit_als: ImplicitALSRetriever,
    content_tfidf: ContentTFIDFRetriever,
    ranker: CatBoostRankerModel,
) -> pd.DataFrame:
    """
    Формирует ранжированный candidate set для warm-пользователя.

    Args:
        user_id: Идентификатор пользователя.
        known_anime_ids: Известные пользователю anime.
        candidate_count: Число кандидатов от каждого retriever-а.
        popularity: Popularity retriever.
        explicit_svd: Explicit SVD retriever.
        implicit_als: Implicit ALS retriever.
        content_tfidf: TF-IDF content retriever.
        ranker: Обученный CatBoost ranker.

    Returns:
        Кандидатов, отсортированных по score ranker-а.
    """
    features = build_ranker_features_for_user(
        user_id=user_id,
        known_anime_ids=known_anime_ids,
        candidate_count=candidate_count,
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
    )

    if features.empty:
        return pd.DataFrame(
            {
                "anime_id": pd.Series(dtype="int64"),
                "ranker_score": pd.Series(dtype="float64"),
                "rank": pd.Series(dtype="int32"),
            }
        )

    ranking = features.loc[:, ["anime_id"]].copy()
    ranking["ranker_score"] = ranker.predict(features)

    ranking = ranking.sort_values(
        ["ranker_score", "anime_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    ranking["rank"] = np.arange(
        1,
        len(ranking) + 1,
        dtype=np.int32,
    )

    return ranking
