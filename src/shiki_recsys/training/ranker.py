from collections.abc import Iterable, Mapping

import pandas as pd

from shiki_recsys.ranking.candidates import build_ranker_features_for_user
from shiki_recsys.ranking.targets import attach_ranker_target
from shiki_recsys.retrievers.content_tfidf import ContentTFIDFRetriever
from shiki_recsys.retrievers.explicit_svd import ExplicitSVDRetriever
from shiki_recsys.retrievers.implicit_als import ImplicitALSRetriever
from shiki_recsys.retrievers.popularity import PopularityRetriever


def build_ranker_training_data(
    *,
    user_ids: Iterable[int],
    known_items_by_user: Mapping[int, set[int]],
    positive_items_by_user: Mapping[int, set[int]],
    candidate_count: int,
    popularity: PopularityRetriever,
    explicit_svd: ExplicitSVDRetriever,
    implicit_als: ImplicitALSRetriever,
    content_tfidf: ContentTFIDFRetriever,
) -> pd.DataFrame:
    """
    Формирует обучающие данные для ranker-а.

    Args:
        user_ids: Пользователи для построения выборки.
        known_items_by_user: Известные пользователям anime.
        positive_items_by_user: Положительные target anime.
        candidate_count: Число кандидатов от каждого retriever-а.
        popularity: Popularity retriever.
        explicit_svd: Explicit SVD retriever.
        implicit_als: Implicit ALS retriever.
        content_tfidf: TF-IDF content retriever.

    Returns:
        Размеченные признаки кандидатов всех пользователей.
    """
    user_frames = []

    for user_id in user_ids:
        features = build_ranker_features_for_user(
            user_id=user_id,
            known_anime_ids=known_items_by_user.get(user_id, set()),
            candidate_count=candidate_count,
            popularity=popularity,
            explicit_svd=explicit_svd,
            implicit_als=implicit_als,
            content_tfidf=content_tfidf,
        )

        user_frames.append(
            attach_ranker_target(
                features,
                positive_items_by_user,
            )
        )

    return pd.concat(
        user_frames,
        ignore_index=True,
    )
