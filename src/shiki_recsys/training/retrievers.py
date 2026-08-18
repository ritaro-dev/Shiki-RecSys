import pandas as pd

from shiki_recsys.config.training import TrainingConfig
from shiki_recsys.features.als_signals import (
    build_als_signed_interactions,
)
from shiki_recsys.features.content_items import (
    build_content_item_features,
)
from shiki_recsys.features.content_users import (
    build_content_user_profiles,
)
from shiki_recsys.preprocessing.interactions import (
    select_explicit_interactions,
)
from shiki_recsys.retrievers.content_tfidf import (
    ContentTFIDFRetriever,
)
from shiki_recsys.retrievers.explicit_svd import (
    ExplicitSVDRetriever,
)
from shiki_recsys.retrievers.implicit_als import (
    ImplicitALSRetriever,
)
from shiki_recsys.retrievers.popularity import (
    PopularityRetriever,
)


def fit_retrievers(
    *,
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[
    PopularityRetriever,
    ExplicitSVDRetriever,
    ImplicitALSRetriever,
    ContentTFIDFRetriever,
]:
    """
    Fit all candidate retrievers on the provided interactions.

    Args:
        interactions: Prepared training interactions.
        catalog: Prepared anime catalog.
        config: Training configuration.

    Returns:
        Fitted popularity, SVD, ALS, and TF-IDF retrievers.
    """
    positive_rating_threshold = config.target.positive_rating_threshold

    explicit_interactions = select_explicit_interactions(interactions)

    popularity = PopularityRetriever(
        relevance_threshold=positive_rating_threshold,
    ).fit(explicit_interactions)

    svd_config = config.retrievers.explicit_svd

    explicit_svd = ExplicitSVDRetriever(
        min_item_explicit_ratings=(svd_config.min_item_explicit_ratings),
        n_factors=svd_config.n_factors,
        n_epochs=svd_config.n_epochs,
        biased=svd_config.biased,
        learning_rate=svd_config.learning_rate,
        regularization=svd_config.regularization,
        init_mean=svd_config.init_mean,
        init_std_dev=svd_config.init_std_dev,
        random_state=config.random_seed,
    ).fit(explicit_interactions)

    als_config = config.retrievers.implicit_als
    confidences = als_config.signal_confidences

    signed_interactions = build_als_signed_interactions(
        interactions,
        rating_8_10_confidence=confidences.rating_8_10,
        watching_confidence=confidences.watching,
        rewatching_confidence=confidences.rewatching,
        completed_confidence=confidences.completed,
        planned_confidence=confidences.planned,
        on_hold_confidence=confidences.on_hold,
        rating_4_5_confidence=confidences.rating_4_5,
        rating_1_3_confidence=confidences.rating_1_3,
    )

    implicit_als = ImplicitALSRetriever(
        factors=als_config.factors,
        regularization=als_config.regularization,
        alpha=als_config.alpha,
        iterations=als_config.iterations,
        random_state=config.random_seed,
    ).fit(signed_interactions)

    item_features = build_content_item_features(catalog)

    content_config = config.retrievers.content_tfidf

    user_profiles = build_content_user_profiles(
        interactions,
        item_features,
        relevance_threshold=positive_rating_threshold,
        max_positive_items=content_config.max_positive_items,
    )

    content_tfidf = ContentTFIDFRetriever().fit(
        item_features,
        user_profiles,
    )

    return (
        popularity,
        explicit_svd,
        implicit_als,
        content_tfidf,
    )
