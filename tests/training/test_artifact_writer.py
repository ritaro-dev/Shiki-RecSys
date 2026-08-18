import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from shiki_recsys.config.training import TrainingConfig
from shiki_recsys.evaluation.model import ModelEvaluationResult
from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.model_artifacts import (
    ArtifactInferenceConfig,
    ArtifactMetadata,
)
from shiki_recsys.training.artifact_writer import write_model_artifacts


def _metadata(artifact_version: str) -> ArtifactMetadata:
    """Build artifact metadata for writer tests."""
    return ArtifactMetadata(
        artifact_version=artifact_version,
        created_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
        inference=ArtifactInferenceConfig(
            retrieval_k=100,
            positive_rating_threshold=8,
            max_positive_items=50,
        ),
    )


def _training_config() -> Mock:
    """Build a training configuration mock for artifact tests."""
    config = Mock(spec=TrainingConfig)
    config.model_dump.return_value = {
        "random_seed": 42,
        "evaluation": {
            "ranking_k": 20,
        },
    }
    return config


def _evaluation() -> ModelEvaluationResult:
    """Build offline evaluation results for artifact tests."""
    return ModelEvaluationResult(
        ranking_k=20,
        recall_at_k=0.31,
        ndcg_at_k=0.24,
        evaluated_users=742,
    )


def test_write_model_artifacts_creates_version_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write a complete immutable artifact version."""
    bundle = Mock(spec=ModelBundle)
    metadata = _metadata("20260810T120000Z")

    def fake_dump(obj: object, path: Path) -> None:
        path.write_bytes(b"bundle")

    monkeypatch.setattr(
        "shiki_recsys.training.artifact_writer.joblib.dump",
        fake_dump,
    )

    version_dir = write_model_artifacts(
        artifacts_dir=tmp_path,
        bundle=bundle,
        metadata=metadata,
        training_config=_training_config(),
        evaluation=_evaluation(),
    )

    assert version_dir == (tmp_path / "versions" / "20260810T120000Z")
    assert (version_dir / "model_bundle.joblib").read_bytes() == b"bundle"

    metadata_payload = json.loads(
        (version_dir / "metadata.json").read_text(encoding="utf-8")
    )

    assert metadata_payload == {
        "artifact_version": "20260810T120000Z",
        "created_at": "2026-08-10T12:00:00+00:00",
        "inference": {
            "retrieval_k": 100,
            "positive_rating_threshold": 8,
            "max_positive_items": 50,
        },
    }

    training_config_payload = yaml.safe_load(
        (version_dir / "training_config.yaml").read_text(
            encoding="utf-8",
        )
    )

    assert training_config_payload == {
        "random_seed": 42,
        "evaluation": {
            "ranking_k": 20,
        },
    }

    evaluation_payload = json.loads(
        (version_dir / "evaluation.json").read_text(
            encoding="utf-8",
        )
    )

    assert evaluation_payload == {
        "ranking_k": 20,
        "recall_at_k": 0.31,
        "ndcg_at_k": 0.24,
        "evaluated_users": 742,
        "evaluation_protocol": "chronological_holdout",
        "evaluated_before_production_refit": True,
    }


def test_write_model_artifacts_rejects_existing_version(
    tmp_path: Path,
) -> None:
    """Reject overwriting an existing artifact version."""
    version_dir = tmp_path / "versions" / "v1"
    version_dir.mkdir(parents=True)

    with pytest.raises(
        FileExistsError,
        match="v1",
    ):
        write_model_artifacts(
            artifacts_dir=tmp_path,
            bundle=Mock(spec=ModelBundle),
            metadata=_metadata("v1"),
            training_config=_training_config(),
            evaluation=_evaluation(),
        )


def test_write_model_artifacts_does_not_publish_failed_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid publishing a version when artifact writing fails."""

    def fail_dump(obj: object, path: Path) -> None:
        raise RuntimeError("dump failed")

    monkeypatch.setattr(
        "shiki_recsys.training.artifact_writer.joblib.dump",
        fail_dump,
    )

    with pytest.raises(
        RuntimeError,
        match="dump failed",
    ):
        write_model_artifacts(
            artifacts_dir=tmp_path,
            bundle=Mock(spec=ModelBundle),
            metadata=_metadata("v1"),
            training_config=_training_config(),
            evaluation=_evaluation(),
        )

    assert not (tmp_path / "versions" / "v1").exists()

    versions_dir = tmp_path / "versions"
    assert not list(versions_dir.iterdir())
