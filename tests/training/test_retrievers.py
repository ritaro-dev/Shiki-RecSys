from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
from pandas.testing import assert_frame_equal

import shiki_recsys.training.retrievers as retrievers_module
from shiki_recsys.training.retrievers import fit_retrievers


def _training_config() -> SimpleNamespace:
    """Build a minimal training configuration for orchestration tests."""
    return SimpleNamespace(
        random_seed=42,
        target=SimpleNamespace(
            positive_rating_threshold=8.0,
        ),
        retrievers=SimpleNamespace(
            explicit_svd=SimpleNamespace(
                min_item_explicit_ratings=5,
                n_factors=64,
                n_epochs=15,
                biased=True,
                learning_rate=0.015,
                regularization=0.1,
                init_mean=0.0,
                init_std_dev=0.1,
            ),
            implicit_als=SimpleNamespace(
                signal_confidences=SimpleNamespace(
                    rating_8_10=2.0,
                    watching=1.0,
                    rewatching=1.0,
                    completed=0.5,
                    planned=0.5,
                    on_hold=0.0,
                    rating_4_5=-1.0,
                    rating_1_3=-2.0,
                ),
                factors=96,
                regularization=0.1,
                alpha=1.0,
                iterations=40,
            ),
            content_tfidf=SimpleNamespace(
                max_positive_items=50,
            ),
        ),
    )


def test_fit_retrievers_builds_all_models(monkeypatch) -> None:
    """Verify retriever training orchestration and configuration wiring."""
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "anime_id": [10, 20, 30],
            "rating": [8, 0, 5],
            "status": ["completed", "planned", "completed"],
            "updated_at": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                ]
            ),
        }
    )
    catalog = pd.DataFrame({"anime_id": [10, 20, 30]})

    signed_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
            "confidence": [2.0],
        }
    )
    item_features = object()
    user_profiles = object()

    popularity = MagicMock()
    explicit_svd = MagicMock()
    implicit_als = MagicMock()
    content_tfidf = MagicMock()

    popularity.fit.return_value = popularity
    explicit_svd.fit.return_value = explicit_svd
    implicit_als.fit.return_value = implicit_als
    content_tfidf.fit.return_value = content_tfidf

    popularity_factory = MagicMock(return_value=popularity)
    svd_factory = MagicMock(return_value=explicit_svd)
    als_factory = MagicMock(return_value=implicit_als)
    content_factory = MagicMock(return_value=content_tfidf)

    build_als = MagicMock(return_value=signed_interactions)
    build_items = MagicMock(return_value=item_features)
    build_profiles = MagicMock(return_value=user_profiles)

    monkeypatch.setattr(
        retrievers_module,
        "PopularityRetriever",
        popularity_factory,
    )
    monkeypatch.setattr(
        retrievers_module,
        "ExplicitSVDRetriever",
        svd_factory,
    )
    monkeypatch.setattr(
        retrievers_module,
        "ImplicitALSRetriever",
        als_factory,
    )
    monkeypatch.setattr(
        retrievers_module,
        "ContentTFIDFRetriever",
        content_factory,
    )
    monkeypatch.setattr(
        retrievers_module,
        "build_als_signed_interactions",
        build_als,
    )
    monkeypatch.setattr(
        retrievers_module,
        "build_content_item_features",
        build_items,
    )
    monkeypatch.setattr(
        retrievers_module,
        "build_content_user_profiles",
        build_profiles,
    )

    result = fit_retrievers(
        interactions=interactions,
        catalog=catalog,
        config=_training_config(),
    )

    assert result == (
        popularity,
        explicit_svd,
        implicit_als,
        content_tfidf,
    )

    explicit_interactions = (
        interactions.loc[interactions["rating"] > 0].copy().reset_index(drop=True)
    )

    assert_frame_equal(
        popularity.fit.call_args.args[0],
        explicit_interactions,
    )
    assert_frame_equal(
        explicit_svd.fit.call_args.args[0],
        explicit_interactions,
    )

    build_items.assert_called_once_with(catalog)

    build_profiles.assert_called_once_with(
        interactions,
        item_features,
        relevance_threshold=8.0,
        max_positive_items=50,
    )

    content_tfidf.fit.assert_called_once_with(
        item_features,
        user_profiles,
    )
