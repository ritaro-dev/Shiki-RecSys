import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.model_artifacts import ArtifactMetadata
from shiki_recsys.training.artifact_writer import write_model_artifacts


def test_write_model_artifacts_creates_version_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проверяет сохранение bundle и metadata версии."""
    bundle = Mock(spec=ModelBundle)
    metadata = ArtifactMetadata(
        artifact_version="20260810T120000Z",
        created_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
    )

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
    )

    assert version_dir == (tmp_path / "versions" / "20260810T120000Z")
    assert (version_dir / "model_bundle.joblib").read_bytes() == b"bundle"

    metadata_payload = json.loads(
        (version_dir / "metadata.json").read_text(encoding="utf-8")
    )

    assert metadata_payload == {
        "artifact_version": "20260810T120000Z",
        "created_at": "2026-08-10T12:00:00+00:00",
    }


def test_write_model_artifacts_rejects_existing_version(
    tmp_path: Path,
) -> None:
    """Проверяет запрет перезаписи существующей версии."""
    version_dir = tmp_path / "versions" / "v1"
    version_dir.mkdir(parents=True)

    with pytest.raises(
        FileExistsError,
        match="v1",
    ):
        write_model_artifacts(
            artifacts_dir=tmp_path,
            bundle=Mock(spec=ModelBundle),
            metadata=ArtifactMetadata(
                artifact_version="v1",
                created_at=datetime.now(UTC),
            ),
        )


def test_write_model_artifacts_does_not_publish_failed_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проверяет отсутствие версии при ошибке сохранения."""

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
            metadata=ArtifactMetadata(
                artifact_version="v1",
                created_at=datetime.now(UTC),
            ),
        )

    assert not (tmp_path / "versions" / "v1").exists()

    versions_dir = tmp_path / "versions"
    assert not list(versions_dir.iterdir())
