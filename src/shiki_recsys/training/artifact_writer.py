import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import yaml

from shiki_recsys.config.training import TrainingConfig
from shiki_recsys.evaluation.model import ModelEvaluationResult
from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.model_artifacts import ArtifactMetadata


def write_model_artifacts(
    *,
    artifacts_dir: Path,
    bundle: ModelBundle,
    metadata: ArtifactMetadata,
    training_config: TrainingConfig,
    evaluation: ModelEvaluationResult,
) -> Path:
    """
    Write an immutable model artifact version.

    Args:
        artifacts_dir: Root artifact directory.
        bundle: Production model bundle.
        metadata: Artifact version metadata.
        training_config: Configuration used for training and evaluation.
        evaluation: Offline evaluation results before production refit.

    Returns:
        Path to the written artifact version.

    Raises:
        FileExistsError: If the artifact version already exists.
    """
    versions_dir = artifacts_dir / "versions"
    version_dir = versions_dir / metadata.artifact_version

    if version_dir.exists():
        raise FileExistsError(
            f"Artifact version уже существует: {metadata.artifact_version}."
        )

    versions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        dir=versions_dir,
        prefix=f".{metadata.artifact_version}-",
    ) as temp_dir:
        temp_path = Path(temp_dir)

        joblib.dump(
            bundle,
            temp_path / "model_bundle.joblib",
        )

        metadata_payload = {
            "artifact_version": metadata.artifact_version,
            "created_at": metadata.created_at.isoformat(),
            "inference": {
                "retrieval_k": metadata.inference.retrieval_k,
                "positive_rating_threshold": metadata.inference.positive_rating_threshold,
                "max_positive_items": metadata.inference.max_positive_items,
            },
        }

        (temp_path / "metadata.json").write_text(
            json.dumps(
                metadata_payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        training_config_payload = training_config.model_dump(
            mode="json",
        )

        (temp_path / "training_config.yaml").write_text(
            yaml.safe_dump(
                training_config_payload,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        evaluation_payload = {
            "ranking_k": evaluation.ranking_k,
            "recall_at_k": evaluation.recall_at_k,
            "ndcg_at_k": evaluation.ndcg_at_k,
            "evaluated_users": evaluation.evaluated_users,
            "evaluation_protocol": "chronological_holdout",
            "evaluated_before_production_refit": True,
        }

        (temp_path / "evaluation.json").write_text(
            json.dumps(
                evaluation_payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        temp_path.rename(version_dir)

    return version_dir
