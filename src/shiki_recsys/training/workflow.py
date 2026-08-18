from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from shiki_recsys.config.training import TrainingConfig
from shiki_recsys.evaluation.model import (
    ModelEvaluationResult,
    evaluate_model_bundle,
)
from shiki_recsys.model_artifacts import (
    ArtifactInferenceConfig,
    ArtifactMetadata,
)
from shiki_recsys.training.artifact_writer import (
    write_model_artifacts,
)
from shiki_recsys.training.pipeline import (
    train_evaluation_bundle,
    train_production_bundle,
)


@dataclass(frozen=True)
class TrainingRunResult:
    """Store the outputs of a complete offline training run."""

    artifact_path: Path
    evaluation: ModelEvaluationResult


def run_training_workflow(
    *,
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    config: TrainingConfig,
    artifacts_dir: Path,
    artifact_version: str,
    created_at: datetime,
) -> TrainingRunResult:
    """
    Train, evaluate, and persist a production model artifact.

    Args:
        interactions: Prepared user interactions.
        catalog: Prepared anime catalog.
        config: Training configuration.
        artifacts_dir: Root artifact directory.
        artifact_version: Immutable artifact version identifier.
        created_at: Artifact creation timestamp.

    Returns:
        Written artifact path and offline evaluation results.
    """
    evaluation_training = train_evaluation_bundle(
        interactions=interactions,
        catalog=catalog,
        config=config,
    )

    evaluation = evaluate_model_bundle(
        bundle=evaluation_training.bundle,
        split=evaluation_training.split,
        config=config,
    )

    production_bundle = train_production_bundle(
        split=evaluation_training.split,
        catalog=catalog,
        config=config,
    )

    metadata = ArtifactMetadata(
        artifact_version=artifact_version,
        created_at=created_at,
        inference=ArtifactInferenceConfig(
            retrieval_k=config.candidate_generation.retrieval_k,
            positive_rating_threshold=(config.target.positive_rating_threshold),
            max_positive_items=(config.retrievers.content_tfidf.max_positive_items),
        ),
    )

    artifact_path = write_model_artifacts(
        artifacts_dir=artifacts_dir,
        bundle=production_bundle,
        metadata=metadata,
        training_config=config,
        evaluation=evaluation,
    )

    return TrainingRunResult(
        artifact_path=artifact_path,
        evaluation=evaluation,
    )
