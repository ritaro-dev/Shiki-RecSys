import json
from datetime import datetime
from pathlib import Path

import joblib

from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.model_artifacts import (
    ArtifactInferenceConfig,
    ArtifactMetadata,
)


def load_model_artifacts(
    *,
    artifacts_dir: Path,
    artifact_version: str,
) -> tuple[ModelBundle, ArtifactMetadata]:
    """
    Загружает конкретную версию model artifacts.

    Args:
        artifacts_dir: Корневая директория artifacts.
        artifact_version: Версия model artifacts.

    Returns:
        Model bundle и metadata версии.

    Raises:
        FileNotFoundError: Если версия или её файлы отсутствуют.
        ValueError: Если metadata не соответствует версии.
    """
    version_dir = artifacts_dir / "versions" / artifact_version

    if not version_dir.is_dir():
        raise FileNotFoundError(f"Artifact version не существует: {artifact_version}.")

    bundle_path = version_dir / "model_bundle.joblib"
    metadata_path = version_dir / "metadata.json"

    if not bundle_path.is_file():
        raise FileNotFoundError(f"Не найден model bundle: {bundle_path}.")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Не найдена artifact metadata: {metadata_path}.")

    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata = ArtifactMetadata(
        artifact_version=metadata_payload["artifact_version"],
        created_at=datetime.fromisoformat(metadata_payload["created_at"]),
        inference=ArtifactInferenceConfig(
            retrieval_k=metadata_payload["inference"]["retrieval_k"],
            positive_rating_threshold=metadata_payload["inference"][
                "positive_rating_threshold"
            ],
            max_positive_items=metadata_payload["inference"]["max_positive_items"],
        ),
    )

    if metadata.artifact_version != artifact_version:
        raise ValueError("Версия в metadata не соответствует директории artifacts.")

    bundle = joblib.load(bundle_path)

    return bundle, metadata


def load_current_model_artifacts(
    *,
    artifacts_dir: Path,
) -> tuple[ModelBundle, ArtifactMetadata]:
    """
    Загружает текущую версию model artifacts.

    Args:
        artifacts_dir: Корневая директория artifacts.

    Returns:
        Current model bundle и metadata.

    Raises:
        FileNotFoundError: Если current не существует.
        ValueError: Если current не содержит версию.
    """
    current_path = artifacts_dir / "current"

    if not current_path.is_file():
        raise FileNotFoundError("Не найден artifacts/current.")

    artifact_version = current_path.read_text(
        encoding="utf-8",
    ).strip()

    if not artifact_version:
        raise ValueError("artifacts/current не содержит версию.")

    return load_model_artifacts(
        artifacts_dir=artifacts_dir,
        artifact_version=artifact_version,
    )
