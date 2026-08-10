from pathlib import Path

import pytest

from shiki_recsys.training.model_promotion import promote_model_version


def test_promote_model_version_sets_current_version(
    tmp_path: Path,
) -> None:
    """Проверяет публикацию сохранённой версии."""
    version_dir = tmp_path / "versions" / "v1"
    version_dir.mkdir(parents=True)

    promote_model_version(
        artifacts_dir=tmp_path,
        artifact_version="v1",
    )

    assert (tmp_path / "current").read_text(encoding="utf-8") == "v1"
    assert not (tmp_path / ".current.tmp").exists()


def test_promote_model_version_replaces_current_version(
    tmp_path: Path,
) -> None:
    """Проверяет переключение current на новую версию."""
    (tmp_path / "versions" / "v1").mkdir(parents=True)
    (tmp_path / "versions" / "v2").mkdir(parents=True)
    (tmp_path / "current").write_text(
        "v1",
        encoding="utf-8",
    )

    promote_model_version(
        artifacts_dir=tmp_path,
        artifact_version="v2",
    )

    assert (tmp_path / "current").read_text(encoding="utf-8") == "v2"


def test_promote_model_version_rejects_missing_version(
    tmp_path: Path,
) -> None:
    """Проверяет запрет публикации отсутствующей версии."""
    with pytest.raises(
        FileNotFoundError,
        match="v1",
    ):
        promote_model_version(
            artifacts_dir=tmp_path,
            artifact_version="v1",
        )

    assert not (tmp_path / "current").exists()
