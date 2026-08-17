from collections.abc import Iterable, Mapping

import pandas as pd

from shiki_recsys.config.training import TrainingConfig
from shiki_recsys.evaluation.ground_truth import (
    build_positive_items_by_user,
)
from shiki_recsys.preprocessing.known_items import build_known_items
from shiki_recsys.ranking.candidates import build_ranker_features_for_user
from shiki_recsys.ranking.catboost import CatBoostRankerModel
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
    Build labeled candidate data for ranker training.

    Args:
        user_ids: Users included in the training dataset.
        known_items_by_user: Anime already known to each user.
        positive_items_by_user: Positive target anime for each user.
        candidate_count: Number of candidates from each retriever.
        popularity: Popularity retriever.
        explicit_svd: Explicit SVD retriever.
        implicit_als: Implicit ALS retriever.
        content_tfidf: TF-IDF content retriever.

    Returns:
        Labeled candidate features for all requested users.
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


def fit_ranker(
    *,
    history_interactions: pd.DataFrame,
    target_interactions: pd.DataFrame,
    config: TrainingConfig,
    popularity: PopularityRetriever,
    explicit_svd: ExplicitSVDRetriever,
    implicit_als: ImplicitALSRetriever,
    content_tfidf: ContentTFIDFRetriever,
) -> CatBoostRankerModel:
    """
    Fit the ranking model on future interactions.

    Args:
        history_interactions: Interactions known when candidates are generated.
        target_interactions: Future interactions used as ranking targets.
        config: Training configuration.
        popularity: Fitted popularity retriever.
        explicit_svd: Fitted explicit SVD retriever.
        implicit_als: Fitted implicit ALS retriever.
        content_tfidf: Fitted TF-IDF retriever.

    Returns:
        Fitted CatBoost ranking model.

    Raises:
        ValueError: If target interactions contain no positive items.
    """
    known_items_by_user = build_known_items(history_interactions)

    positive_items_by_user = build_positive_items_by_user(
        target_interactions,
        positive_rating_threshold=config.target.positive_rating_threshold,
    )

    if not positive_items_by_user:
        raise ValueError("Target interactions contain no positive items.")

    training_data = build_ranker_training_data(
        user_ids=positive_items_by_user,
        known_items_by_user=known_items_by_user,
        positive_items_by_user=positive_items_by_user,
        candidate_count=config.candidate_generation.retrieval_k,
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
    )

    ranker = CatBoostRankerModel(
        config.ranker,
        random_seed=config.random_seed,
    )
    ranker.fit(training_data)

    return ranker
