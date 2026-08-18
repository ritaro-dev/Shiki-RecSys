from dataclasses import dataclass

import pandas as pd

from shiki_recsys.config.training import TrainingConfig
from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.preprocessing.splitting import (
    InteractionSplit,
    chronological_split,
)
from shiki_recsys.training.ranker import fit_ranker
from shiki_recsys.training.retrievers import fit_retrievers


@dataclass(frozen=True)
class EvaluationTrainingResult:
    """Store the evaluation model bundle and chronological data split."""

    bundle: ModelBundle
    split: InteractionSplit


def train_evaluation_bundle(
    *,
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    config: TrainingConfig,
) -> EvaluationTrainingResult:
    """
    Train a model bundle for unbiased offline evaluation.

    Args:
        interactions: Prepared user interactions.
        catalog: Prepared anime catalog.
        config: Training configuration.

    Returns:
        Evaluation model bundle and chronological data split.
    """
    split_config = config.dataset.split

    split = chronological_split(
        interactions,
        validation_fraction=split_config.validation_fraction,
        test_fraction=split_config.test_fraction,
        min_interactions_per_user=split_config.min_interactions_per_user,
    )

    (
        train_popularity,
        train_explicit_svd,
        train_implicit_als,
        train_content_tfidf,
    ) = fit_retrievers(
        interactions=split.train,
        catalog=catalog,
        config=config,
    )

    ranker = fit_ranker(
        history_interactions=split.train,
        target_interactions=split.validation,
        config=config,
        popularity=train_popularity,
        explicit_svd=train_explicit_svd,
        implicit_als=train_implicit_als,
        content_tfidf=train_content_tfidf,
    )

    evaluation_history = pd.concat(
        [
            split.train,
            split.validation,
        ],
        ignore_index=True,
    )

    (
        popularity,
        explicit_svd,
        implicit_als,
        content_tfidf,
    ) = fit_retrievers(
        interactions=evaluation_history,
        catalog=catalog,
        config=config,
    )

    bundle = ModelBundle(
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
        ranker=ranker,
    )

    return EvaluationTrainingResult(
        bundle=bundle,
        split=split,
    )


def train_production_bundle(
    *,
    split: InteractionSplit,
    catalog: pd.DataFrame,
    config: TrainingConfig,
) -> ModelBundle:
    """
    Train the production model bundle using all available interactions.

    Args:
        split: Chronological split previously used for evaluation.
        catalog: Prepared anime catalog.
        config: Training configuration.

    Returns:
        Production model bundle.
    """
    ranker_history = pd.concat(
        [
            split.train,
            split.validation,
        ],
        ignore_index=True,
    )

    (
        ranker_popularity,
        ranker_explicit_svd,
        ranker_implicit_als,
        ranker_content_tfidf,
    ) = fit_retrievers(
        interactions=ranker_history,
        catalog=catalog,
        config=config,
    )

    ranker = fit_ranker(
        history_interactions=ranker_history,
        target_interactions=split.test,
        config=config,
        popularity=ranker_popularity,
        explicit_svd=ranker_explicit_svd,
        implicit_als=ranker_implicit_als,
        content_tfidf=ranker_content_tfidf,
    )

    all_interactions = pd.concat(
        [
            split.train,
            split.validation,
            split.test,
        ],
        ignore_index=True,
    )

    (
        popularity,
        explicit_svd,
        implicit_als,
        content_tfidf,
    ) = fit_retrievers(
        interactions=all_interactions,
        catalog=catalog,
        config=config,
    )

    return ModelBundle(
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
        ranker=ranker,
    )
