import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib

from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.model_artifacts import ArtifactMetadata


def write_model_artifacts(
    *,
    artifacts_dir: Path,
    bundle: ModelBundle,
    metadata: ArtifactMetadata,
) -> Path:
    """
    Сохраняет immutable-версию model artifacts.

    Args:
        artifacts_dir: Корневая директория artifacts.
        bundle: Согласованный набор моделей.
        metadata: Metadata версии artifacts.

    Returns:
        Путь к сохранённой версии.

    Raises:
        FileExistsError: Если версия уже существует.
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
        }

        (temp_path / "metadata.json").write_text(
            json.dumps(
                metadata_payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        temp_path.rename(version_dir)

    return version_dir
