from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
from pandas.testing import assert_frame_equal

import shiki_recsys.training.pipeline as pipeline_module
from shiki_recsys.preprocessing.splitting import InteractionSplit
from shiki_recsys.training.pipeline import (
    train_evaluation_bundle,
    train_production_bundle,
)


def test_train_evaluation_bundle_builds_evaluation_bundle(
    monkeypatch,
) -> None:
    """Verify the complete model training orchestration."""
    train = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
        }
    )
    validation = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [20],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [30],
        }
    )

    split = InteractionSplit(
        train=train,
        validation=validation,
        test=test,
        evaluation_user_ids=frozenset({1}),
        train_only_user_ids=frozenset(),
    )

    config = SimpleNamespace(
        dataset=SimpleNamespace(
            split=SimpleNamespace(
                validation_fraction=0.1,
                test_fraction=0.1,
                min_interactions_per_user=10,
            )
        )
    )

    initial_retrievers = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    final_retrievers = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    ranker = MagicMock()

    chronological_split = MagicMock(return_value=split)
    fit_retrievers = MagicMock(
        side_effect=[
            initial_retrievers,
            final_retrievers,
        ]
    )
    fit_ranker = MagicMock(return_value=ranker)

    monkeypatch.setattr(
        pipeline_module,
        "chronological_split",
        chronological_split,
    )
    monkeypatch.setattr(
        pipeline_module,
        "fit_retrievers",
        fit_retrievers,
    )
    monkeypatch.setattr(
        pipeline_module,
        "fit_ranker",
        fit_ranker,
    )

    catalog = pd.DataFrame(
        {
            "anime_id": [10, 20, 30],
        }
    )
    interactions = pd.concat(
        [
            train,
            validation,
            test,
        ],
        ignore_index=True,
    )

    result = train_evaluation_bundle(
        interactions=interactions,
        catalog=catalog,
        config=config,
    )

    chronological_split.assert_called_once_with(
        interactions,
        validation_fraction=0.1,
        test_fraction=0.1,
        min_interactions_per_user=10,
    )

    first_fit = fit_retrievers.call_args_list[0]
    assert_frame_equal(
        first_fit.kwargs["interactions"],
        train,
    )

    fit_ranker.assert_called_once_with(
        history_interactions=train,
        target_interactions=validation,
        config=config,
        popularity=initial_retrievers[0],
        explicit_svd=initial_retrievers[1],
        implicit_als=initial_retrievers[2],
        content_tfidf=initial_retrievers[3],
    )

    final_interactions = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    second_fit = fit_retrievers.call_args_list[1]
    assert_frame_equal(
        second_fit.kwargs["interactions"],
        final_interactions,
    )

    assert result.bundle.popularity is final_retrievers[0]
    assert result.bundle.explicit_svd is final_retrievers[1]
    assert result.bundle.implicit_als is final_retrievers[2]
    assert result.bundle.content_tfidf is final_retrievers[3]
    assert result.bundle.ranker is ranker
    assert result.split is split


def test_train_production_bundle_uses_all_interactions(
    monkeypatch,
) -> None:
    """Verify production training uses test targets and all interactions."""
    train = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
        }
    )
    validation = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [20],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [30],
        }
    )

    split = InteractionSplit(
        train=train,
        validation=validation,
        test=test,
        evaluation_user_ids=frozenset({1}),
        train_only_user_ids=frozenset(),
    )

    config = MagicMock()
    catalog = pd.DataFrame({"anime_id": [10, 20, 30]})

    ranker_retrievers = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    final_retrievers = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    ranker = MagicMock()

    fit_retrievers = MagicMock(
        side_effect=[
            ranker_retrievers,
            final_retrievers,
        ]
    )
    fit_ranker = MagicMock(return_value=ranker)

    monkeypatch.setattr(
        pipeline_module,
        "fit_retrievers",
        fit_retrievers,
    )
    monkeypatch.setattr(
        pipeline_module,
        "fit_ranker",
        fit_ranker,
    )

    bundle = train_production_bundle(
        split=split,
        catalog=catalog,
        config=config,
    )

    ranker_history = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    first_fit = fit_retrievers.call_args_list[0]

    assert_frame_equal(
        first_fit.kwargs["interactions"],
        ranker_history,
    )

    fit_ranker.assert_called_once()

    ranker_call = fit_ranker.call_args

    assert_frame_equal(
        ranker_call.kwargs["history_interactions"],
        ranker_history,
    )
    assert_frame_equal(
        ranker_call.kwargs["target_interactions"],
        test,
    )

    assert ranker_call.kwargs["config"] is config
    assert ranker_call.kwargs["popularity"] is ranker_retrievers[0]
    assert ranker_call.kwargs["explicit_svd"] is ranker_retrievers[1]
    assert ranker_call.kwargs["implicit_als"] is ranker_retrievers[2]
    assert ranker_call.kwargs["content_tfidf"] is ranker_retrievers[3]

    all_interactions = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    second_fit = fit_retrievers.call_args_list[1]

    assert_frame_equal(
        second_fit.kwargs["interactions"],
        all_interactions,
    )

    assert bundle.popularity is final_retrievers[0]
    assert bundle.explicit_svd is final_retrievers[1]
    assert bundle.implicit_als is final_retrievers[2]
    assert bundle.content_tfidf is final_retrievers[3]
    assert bundle.ranker is ranker
